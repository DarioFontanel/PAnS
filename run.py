import utils
import argparser
import os
from utils.logger import Logger
from dataset.utils import round2nearest_multiple
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

import numpy as np
import random
import torch
from torch.utils import data
from torch import distributed

from dataset import StreetHazardsSegmentation
from dataset import transform
from metrics import StreamSegMetrics

from segmentation_module import make_model

from train import Trainer

def save_ckpt(path, model, trainer, optimizer, scheduler, epoch, best_score):
    """ save current model
    """
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_score": best_score,
        "trainer_state": trainer.state_dict()
    }

    torch.save(state, path)

def get_dataset(opts):
    """ Dataset And Augmentation
    """
    resize_scales = [300 / 720, 375 / 720, 450 / 720, 525 / 720, 1000 / 1280]

    # TRAIN
    basic_transform = transform.Compose([
        transform.ToTensor(),
        transform.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    if opts.dataset == 'streethazards':
        train_transform = transform.Compose([
            transform.RandomScale([resize_scales[0], resize_scales[-1]]),
            transform.RandomCrop(opts.crop_size, pad_if_needed=True), # 450
            transform.RandomHorizontalFlip(),
            basic_transform
        ])
    else:
        raise NotImplementedError(f"Transformations for /{opts.dataset}/ dataset not available")

    # VALIDATION
    val_transform = basic_transform

    # TEST
    heights = [int(720 * s) for s in resize_scales]
    widhts = [int(1280 * s) for s in resize_scales]

    widhts = round2nearest_multiple(widhts, 8)
    heights = round2nearest_multiple(heights, 8)

    if opts.multi_scala:
        if opts.dataset == 'streethazards':
            test_transform = []
            for height, widht in zip(heights, widhts):
                test_transform.append(transform.Compose([
                    transform.Resize((height, widht)),
                    basic_transform
                ]))
        else:
            raise NotImplementedError(f"Multi scale transformations for /{opts.dataset}/ dataset not available")

    else:
        if opts.dataset == 'streethazards':
            test_transform = transform.Compose([
                        transform.Resize((heights[-1], widhts[-1])),
                        basic_transform
                    ])
        else:
            raise NotImplementedError(f"Transformations for /{opts.dataset}/ dataset not available")

    # DATATSET
    if opts.dataset == 'streethazards':
        train_dataset = StreetHazardsSegmentation
        test_dataset = StreetHazardsSegmentation
    else:
        raise NotImplementedError(f"/{opts.dataset}/ dataset not available")

    train_dst = train_dataset(root=opts.data_root, split='train', transform=train_transform,
                        basic_transform=basic_transform)


    val_dst = train_dataset(root=opts.data_root, split='validation', transform=val_transform,
                      basic_transform=basic_transform)

    test_dst = test_dataset(root=opts.data_root, split='test', transform=test_transform, basic_transform=basic_transform,
                       multiple_resizes_test=opts.multi_scala)

    return train_dst, val_dst, test_dst


def main(opts):
    distributed.init_process_group(backend='nccl', init_method='env://')
    device_id, device = opts.local_rank, torch.device(opts.local_rank)
    rank, world_size = distributed.get_rank(), distributed.get_world_size()
    torch.cuda.set_device(device_id)
    # Initialize logging
    logdir_full = f"{opts.logdir}/{opts.dataset}/{opts.name}/"
    if rank == 0:
        logger = Logger(logdir_full, rank=rank, summary=opts.visualize)
    else:
        logger = Logger(logdir_full, rank=rank, summary=False)

    logger.print(f"Device: {device}")

    checkpoint_path = f"checkpoints/{opts.dataset}/{opts.name}.pth"
    os.makedirs(f"checkpoints/{opts.dataset}", exist_ok=True)

    # Set up random seed
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    train_dst, val_dst, test_dst = get_dataset(opts)

    train_loader = data.DataLoader(train_dst, batch_size=opts.batch_size,
                                   sampler=DistributedSampler(train_dst, num_replicas=world_size, rank=rank),
                                   num_workers=opts.num_workers, drop_last=True)
    val_loader = data.DataLoader(val_dst, batch_size=1,
                                 sampler=DistributedSampler(val_dst, num_replicas=world_size, rank=rank),
                                 num_workers=opts.num_workers, drop_last=True)
    logger.info(f"Dataset: {opts.dataset}, Train set: {len(train_dst)}, Val set (w/o anomalies): {len(val_dst)},"
                f" Test set (w/ anomalies): {len(test_dst)}, #training classes: {opts.num_classes}")
    logger.info(f"Total batch size is {opts.batch_size * world_size}")
    opts.n_gpus = world_size

    # xxx Set up model
    logger.info(f"Backbone: {opts.backbone}")

    model = make_model(opts)

    logger.info(f"[!] Model made with{'out' if opts.no_pretrained else ''} pre-trained")

    # xxx Set up optimizer
    params = []
    params.append({"params": filter(lambda p: p.requires_grad, model.body.parameters()),
                   'weight_decay': opts.weight_decay})
    params.append({"params": filter(lambda p: p.requires_grad, model.head.parameters()),
                   'weight_decay': opts.weight_decay})
    params.append({"params": filter(lambda p: p.requires_grad, model.cls.parameters()),
                   'weight_decay': opts.weight_decay})

    optimizer = torch.optim.SGD(params, lr=opts.lr, momentum=0.9, nesterov=True)

    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, max_iters=opts.epochs * len(train_loader), power=opts.lr_power)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    else:
        raise NotImplementedError

    model = model.to(device)

    # Put the model on GPU
    model = DistributedDataParallel(model, device_ids=[opts.local_rank], output_device=opts.local_rank)

    trainer = Trainer(model, device=device, opts=opts)

    # xxx Handle checkpoint for current model (model old will always be as previous step or None)
    best_score = 0.0
    cur_epoch = 0
    if opts.ckpt is not None:
        ckpt_path = f"checkpoints/{opts.dataset}/{opts.ckpt}"
        assert os.path.isfile(ckpt_path), "Error, ckpt not found. Check the correct directory"
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        cur_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint['best_score']
        logger.info("[!] Model restored from %s" % ckpt_path)
        # if we want to resume training, resume trainer from checkpoint
        if 'trainer_state' in checkpoint:
            trainer.load_state_dict(checkpoint['trainer_state'])
        del checkpoint
    else:
        logger.info("[!] Train from scratch")

    # xxx Train procedure
    # print opts before starting training to log all parameters
    logger.add_table("Opts", vars(opts))

    # For visualization -> select random batches to display on tensorboard
    if rank == 0 and opts.sample_num > 0:
        sample_ids = np.random.choice(len(val_loader), opts.sample_num, replace=False)  # sample idxs for visualization
        logger.info(f"The samples id are {sample_ids}")
    else:
        sample_ids = None

    label2color = utils.Label2Color(cmap=utils.color_map(opts.dataset))  # convert labels t   o images
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])  # de-normalization for original images

    val_metrics = StreamSegMetrics(opts.num_classes, opts.unk_class)
    val_score = None
    results = {}

    # check if random is equal here.
    logger.print(torch.randint(0, 100, (1, 1)))
    # train/val here
    while cur_epoch < opts.epochs and not opts.test:
        # =====  Train  =====
        model.train()

        epoch_loss = trainer.train(cur_epoch=cur_epoch, optim=optimizer, train_loader=train_loader,
                                                                               scheduler=scheduler, logger=logger)

        logger.info(f"End of Epoch {cur_epoch}/{opts.epochs -1}, Class Loss={epoch_loss},")

        # =====  Log metrics on Tensorboard =====
        logger.add_scalar("E-Loss/E-Loss", epoch_loss, cur_epoch)

        # =====  Validation  =====

        if (cur_epoch + 1) % opts.val_interval == 0:
            logger.info("validate on val set...")
            model.eval()
            val_loss, val_score, ret_samples = trainer.validate(loader=val_loader, metrics=val_metrics,
                                                                logger=logger, ret_samples_ids=sample_ids)

            logger.print("Done validation")
            logger.info(f"End of Validation {cur_epoch}/{opts.epochs}, Validation Loss={val_loss}")

            logger.info(val_metrics.to_str(val_score))

            # =====  Log metrics on Tensorboard =====
            # visualize validation score and samples
            logger.add_scalar("V-Loss", val_loss, cur_epoch)
            logger.add_scalar("Val_Overall_Acc", val_score['Overall Acc'], cur_epoch)
            logger.add_scalar("Val_MeanIoU", val_score['Mean IoU'], cur_epoch)
            logger.add_table("Val_Class_IoU", val_score['Class IoU'], cur_epoch)
            logger.add_table("Val_Acc_IoU", val_score['Class Acc'], cur_epoch)
            logger.add_figure("Val_Confusion_Matrix", val_score['Confusion Matrix'], cur_epoch)

            for k, (img, target) in enumerate(ret_samples):
                img = (denorm(img) * 255).astype(np.uint8)
                target = label2color(target).transpose(2, 0, 1).astype(np.uint8)

                concat_img = np.concatenate((img, target), axis=2)  # concat along width
                logger.add_image(f'Validation_sample_{k}', concat_img, cur_epoch)

            # keep the metric to print them at the end of training
            results["V-IoU"] = val_score['Class IoU']
            results["V-Acc"] = val_score['Class Acc']

        # =====  Save Model  =====
        if rank == 0:  # save best model at the last iteration
            score = val_score['Mean IoU'] if val_score is not None else 0.  # use last score we have
            # best model to build incremental steps
            save_ckpt(checkpoint_path, model, trainer, optimizer, scheduler, cur_epoch, score)
            logger.info("[!] Checkpoint saved.")

        cur_epoch += 1

    # =====  Save Best Model at the end of training =====
    if rank == 0 and not opts.test:  # save best model at the last iteration
        # best model to build incremental steps
        save_ckpt(checkpoint_path, model, trainer, optimizer, scheduler, cur_epoch, best_score)
        logger.info("[!] Best model Checkpoint saved.")

    torch.distributed.barrier()

    # xxx From here starts the test code
    logger.info("*** Test the model on all seen classes...")
    # make data loader
    test_loader = data.DataLoader(test_dst, batch_size=1,
                                  sampler=DistributedSampler(test_dst, num_replicas=world_size, rank=rank),
                                  num_workers=opts.num_workers, drop_last=True)

    model = make_model(opts)
    # Put the model on GPU
    model = DistributedDataParallel(model.cuda(device), device_ids=[opts.local_rank], output_device=opts.local_rank)

    if opts.ckpt_test is not None:
        checkpoint_path = f"checkpoints/{opts.dataset}/{opts.ckpt_test}"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    logger.info(f"*** Model restored from {checkpoint_path}")
    del checkpoint

    trainer = Trainer(model, device=device, opts=opts)
    model.eval()

    val_score = trainer.test(loader=test_loader,  metrics=val_metrics)

    # =====  Log test results on Tensorboard =====
    # visualize test score and samples
    logger.print("Done test")
    logger.info(f"*** End of Test")
    logger.info(val_metrics.to_str(val_score))
    logger.add_table("Test_Class_IoU", val_score['Class IoU'])
    logger.add_table("Test_Class_Acc", val_score['Class Acc'])
    logger.add_figure("Test_Confusion_Matrix", val_score['Confusion Matrix'])
    # logger.add_figure("ROC Curve", val_score['ROC_Curve'])
    results["T-IoU"] = val_score['Class IoU']
    results["T-Acc"] = val_score['Class Acc']
    logger.add_results(results)

    logger.add_scalar("T_Overall_Acc", val_score['Overall Acc'])
    logger.add_scalar("T_MeanIoU", val_score['Mean IoU'])
    logger.add_scalar("T_MeanAcc", val_score['Mean Acc'])
    logger.add_scalar("AUROC", val_score['AUROC'])
    logger.add_scalar("AUPR", val_score['AUPR'])
    logger.add_scalar("FPR95", val_score['FPR95'])

    logger.close()


if __name__ == '__main__':
    parser = argparser.get_argparser()

    opts = parser.parse_args()
    opts = argparser.modify_command_options(opts)

    main(opts)
