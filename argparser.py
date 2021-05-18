import argparse


def modify_command_options(opts):
    if opts.default:
        opts.multi_scala = True
        opts.num_classes = 13
        opts.fix_bn = False
        opts.lr = 0.007
        opts.momentum = 0.9
        opts.weight_decay = 1e-4
        opts.deepsup = 0.4
        opts.backbone = 'resnet50'
        opts.head = 'PPM'
        opts.crop_size = 450
        opts.num_workers = 16
        opts.batch_size = 2
        opts.epochs = 40
        opts.classifier = "cosine"
        opts.cosine_scores = True

    if not opts.visualize:
        opts.sample_num = 0

    return opts


def get_argparser():
    parser = argparse.ArgumentParser()

    # Performance Options
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--random_seed", type=int, default=42,
                        help="random seed (default: 42)")
    parser.add_argument("--num_workers", type=int, default=16,
                        help='number of workers (default: 16)')

    # Datset Options
    parser.add_argument("--data_root", type=str, default='data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='streethazards',
                        choices=['streethazards'], help='Which training dataset to use')

    parser.add_argument("--num_classes", type=int, default=13,
                        help="num classes (default: 13)")
    parser.add_argument("--unk_class", type=int, default=13,
                        help="unknown class to segment")

    # Train Options
    parser.add_argument("--epochs", type=int, default=40,
                        help="epoch number (default: 40)")
    parser.add_argument("--fix_bn", action='store_true', default=False,
                        help='fix batch normalization during training (default: False)')

    parser.add_argument("--batch_size", type=int, default=2,
                        help='batch size (default: 2)')
    parser.add_argument("--crop_size", type=int, default=512,
                        help="crop size (default: 512)")

    parser.add_argument("--lr", type=float, default=0.007,
                        help="learning rate (default: 0.007)")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help='momentum for SGD (default: 0.9)')
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')

    parser.add_argument("--lr_policy", type=str, default='poly',
                        choices=['poly', 'step'], help="lr schedule policy (default: poly)")
    parser.add_argument("--lr_decay_step", type=int, default=5000,
                        help="decay step for stepLR (default: 5000)")
    parser.add_argument("--lr_decay_factor", type=float, default=0.1,
                        help="decay factor for stepLR (default: 0.1)")
    parser.add_argument("--lr_power", type=float, default=0.9,
                        help="power for polyLR (default: 0.9)")
    parser.add_argument("--deepsup", type=float, default=0.4,
                        help="Scaling factor for Deepsup loss (default: 0.4)")

    # Validation Options
    parser.add_argument("--val_on_trainset", action='store_true', default=False,
                        help="enable validation on train set (default: False)")
    parser.add_argument("--multi_scala", action='store_true', default=False,
                        help='whether to use or not multiple resizes for testing (default: True)')

    # Logging Options
    parser.add_argument("--logdir", type=str, default='./logs',
                        help="path to Log directory (default: ./logs)")
    parser.add_argument("--name", type=str, default='Experiment',
                        help="name of the experiment - to append to log directory (default: Experiment)")
    parser.add_argument("--sample_num", type=int, default=6,
                        help='number of samples for visualization (default: 8)')
    parser.add_argument("--visualize",  action='store_false', default=True,
                        help="visualization on tensorboard (def: Yes)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=1,
                        help="epoch interval for eval (default: 1)")
    parser.add_argument("--ckpt_interval", type=int, default=1,
                        help="epoch interval for saving model (default: 1)")

    # Model Options
    parser.add_argument("--backbone", type=str, default='resnet50',
                        choices=['resnet50', 'resnet101'], help='backbone for the body (def: resnet50)')
    parser.add_argument("--output_stride", type=int, default=8,
                        choices=[8, 16], help='stride for the backbone (def: 8)')
    parser.add_argument("--no_pretrained", action='store_true', default=False,
                        help='Wheather to use pretrained or not (def: True)')
    parser.add_argument("--norm_act", type=str, default="iabn_sync",
                        choices=['iabn_sync', 'iabn', 'abn', 'std'], help='Which BN to use (def: iabn_sync')
    parser.add_argument("--pooling", type=int, default=32,
                        help='pooling in ASPP for the validation phase (def: 32)')
    parser.add_argument("--head", type=str, default='PPM',
                        choices=['PPM'], help='head to use (def: PPM)')
    parser.add_argument("--classifier", type=str, default='standard',
                        choices=['standard', 'cosine'], help='classifier to use (def: standard)')
    parser.add_argument("--cosine_scores", action='store_true', default=False,
                        help='Wheather to use the direct cosine scores instead of softmax (def: False)')
    parser.add_argument("--msp", action='store_true', default=False,
                        help='Wheather to use the MSP or not (def: False)')


    # Test and Checkpoint options
    parser.add_argument("--test",  action='store_true', default=False,
                        help="Whether to train or test only (def: train and test)")
    parser.add_argument("--ckpt_test", default=None, type=str,
                        help="Name of a specific trained model (starts from checkpoints/{dataset}). Different from the "
                        "name of the exp. loaded when --test is True. If none, it uses the ckpt named as the experiment")
    parser.add_argument("--ckpt", default=None, type=str,
                        help="Name of trained model (it starts from checkpoints/{dataset}). None if retrain the model")
    parser.add_argument('--opt_level', type=str, choices=['O0', 'O1', 'O2', 'O3'], default='O0')

    # Standard protocols
    parser.add_argument('--default', action='store_true', default=False,
                        help="Whether to use default protocol or not")

    return parser
