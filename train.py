import torch
from torch import distributed
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as functional
import skimage.measure as measure


class Trainer:
    def __init__(self, model, device, opts):

        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

        self.deepsup_factor = opts.deepsup

        self.unk_class = opts.unk_class
        self.multi_scala = opts.multi_scala
        self.msp = opts.msp
        self.cosine_scores = opts.cosine_scores

    def train(self, cur_epoch, optim, train_loader, scheduler=None, print_int=10, logger=None):
        """Train and return epoch loss"""
        logger.info("Epoch %d, lr = %f" % (cur_epoch, optim.param_groups[0]['lr']))

        device = self.device
        model = self.model
        criterion = self.criterion

        epoch_loss = 0.0
        interval_loss = 0.0

        train_loader.sampler.set_epoch(cur_epoch)

        model.train()

        for cur_step, (images, labels) in enumerate(train_loader):
            loss_deepsup = torch.zeros(1).to(self.device)

            # normal training images
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            targets = labels.clone()
            optim.zero_grad()

            outputs, outputs_deepsup, feat_head = model(images)

            # xxx Criterion Loss
            loss = criterion(outputs, targets) # B x H x W

            # xxx DeepSup Loss
            if outputs_deepsup is not None:
                loss_deepsup = criterion(outputs_deepsup, targets) * self.deepsup_factor

            loss = loss + loss_deepsup

            loss.backward()

            optim.step()

            if scheduler is not None:
                scheduler.step()

            epoch_loss += loss.item()
            interval_loss += loss.item()

            if (cur_step + 1) % print_int == 0:
                interval_loss = interval_loss / print_int

                logger.info(f"Epoch {cur_epoch}, Batch {cur_step + 1}/{len(train_loader)}," f" Loss={interval_loss}")

                # visualization
                if logger is not None:
                    x = cur_epoch * len(train_loader) + cur_step + 1
                    logger.add_scalar('Loss/Loss', interval_loss, x)
                interval_loss = 0.0

        # collect statistics from multiple processes
        epoch_loss = torch.tensor(epoch_loss).to(self.device)

        torch.distributed.reduce(epoch_loss, dst=0)

        if distributed.get_rank() == 0:
            epoch_loss = epoch_loss / distributed.get_world_size() / len(train_loader)

        logger.info(f"Epoch {cur_epoch}, Class Loss={epoch_loss}")
        return epoch_loss

    def validate(self, loader, metrics, ret_samples_ids=None, logger=None):
        """Do validation and return specified samples"""
        metrics.reset()
        model = self.model
        device = self.device
        criterion = self.criterion
        model.eval()

        class_loss = 0.0

        ret_samples = []
        with torch.no_grad():

            if distributed.get_rank() == 0:
                pbar = tqdm(total=len(loader))
            else:
                pbar = None

            for i, (images, labels) in enumerate(loader):

                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)
                interpolation_size = (labels.shape[1], labels.shape[2])

                outputs, _, _, _ = model(images, body_and_head=True, interpolate=False)
                outputs = functional.interpolate(outputs, size=interpolation_size, mode="bilinear", align_corners=False)

                loss = criterion(outputs, labels)
                class_loss += loss.item()

                outputs = nn.functional.softmax(outputs, dim=1)
                probabilities, prediction = outputs.max(dim=1)

                metrics.update(labels.squeeze(0).cpu().numpy(),
                               prediction.squeeze(0).cpu().numpy(),
                               probabilities.squeeze(0).cpu().numpy())

                if ret_samples_ids is not None and i in ret_samples_ids:  # get samples
                    # resizing for logger
                    visualization_size = (interpolation_size[0] // 2, interpolation_size[1] // 2)

                    images = functional.interpolate(images, size=visualization_size, mode="bilinear", align_corners=False)
                    prediction = functional.interpolate(prediction.unsqueeze(0).to(torch.float), size=visualization_size,
                                                        mode="nearest").squeeze(0).to(torch.long)

                    images = images[0].detach().cpu().numpy()
                    prediction = prediction[0].detach().cpu().numpy()

                    ret_samples.append((images, prediction))

                if pbar is not None:
                    pbar.update(1)

            if pbar is not None:
                pbar.close()

            # collect statistics from multiple processes
            metrics.synch(device)
            score = metrics.get_results()

            class_loss = torch.tensor(class_loss).to(self.device)

            torch.distributed.reduce(class_loss, dst=0)

            if distributed.get_rank() == 0:
                class_loss = class_loss / distributed.get_world_size() / len(loader)

            if logger is not None:
                logger.info(f"Validation, Class Loss={class_loss}")

        return class_loss, score, ret_samples

    def test(self, loader, metrics):
        """Do validation and return specified samples"""
        metrics.reset()
        model = self.model
        device = self.device
        model.eval()

        with torch.no_grad():
            pbar = tqdm(total=len(loader))
            for i, (images_resized_list, labels) in enumerate(loader):

                interpolation_size = (labels.shape[1], labels.shape[2])
                anomaly_scores = torch.zeros(1, 1, labels.shape[1], labels.shape[2])
                scores = torch.zeros(1, model.module.cls.classes, labels.shape[1], labels.shape[2])

                if not self.multi_scala:
                    images_resized_list = [images_resized_list]

                labels = labels.to(device, dtype=torch.long)
                for idx, _ in enumerate(images_resized_list):
                    # original images
                    img = images_resized_list[idx].to(device, dtype=torch.float32)

                    # forward pass
                    logits, _, feat_head, feat_body = model(img, body_and_head=True, interpolate=False)

                    if self.cosine_scores:
                        logits_max, _ = logits.max(dim=1)
                        logits_max = ((logits_max + 10.) / 20.).unsqueeze(0)
                        outputs = functional.interpolate(logits_max, size=interpolation_size, mode="bilinear",
                                                         align_corners=False)
                        anomaly_scores = anomaly_scores + outputs.cpu() / len(images_resized_list)

                    logits = functional.interpolate(logits, size=interpolation_size, mode="bilinear",
                                                    align_corners=False)
                    outputs = nn.functional.softmax(logits, dim=1)
                    scores = scores + outputs.cpu() / len(images_resized_list)

                if self.msp:
                    anomaly_scores = scores

                _, prediction = scores.max(dim=1)
                anomaly_probabilities, _ = anomaly_scores.max(dim=1)

                metrics.update(labels.squeeze(0).cpu().numpy(),
                               prediction.squeeze(0).cpu().numpy(),
                               anomaly_probabilities.squeeze(0).cpu().numpy())

                pbar.update(1)

            # collect statistics from multiple processes
            metrics.synch(device)
            score = metrics.get_results()

        return score

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass
