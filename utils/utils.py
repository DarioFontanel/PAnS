from torchvision.transforms.functional import normalize
import torch.nn as nn
import numpy as np



def denormalize(tensor, mean, std):
    mean = np.array(mean)
    std = np.array(std)

    _mean = -mean/std
    _std = 1/std
    return normalize(tensor, _mean, _std)


class Denormalize(object):
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean/std
        self._std = 1/std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1,1,1)) / self._std.reshape(-1,1,1)
        return normalize(tensor, self._mean, self._std)


def fix_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False


def color_map(dataset):
    if dataset=='streethazards':
        return streethazards_cmap()

def streethazards_cmap():
    # Left: modified labels after -1. Right: original label
    # total classes on which the model is trained: 13 (12 common classes + sky/unlabeled)
    # total classes with anomaly: 14
    cmap = np.zeros((256, 3))

    cmap[255] = [0, 0, 0]       # // padding       =   -1 (n.d.) # padding, to ignore

    cmap[0] = [0,   0,   0]     # // unlabeled     =   0 (1)     # sky and unlabeled (used in training)
    cmap[1] = [70,  70,  70]    # // building      =   1 (2),
    cmap[2] = [190, 153, 153]   # // fence         =   2 (3),
    cmap[3] = [250, 170, 160]   # // other         =   3 (4),    # background
    cmap[4] = [220,  20,  60]   # // pedestrian    =   4 (5),
    cmap[5] = [153, 153, 153]   # // pole          =   5 (6),
    cmap[6] = [157, 234,  50]   # // road line     =   6 (7),
    cmap[7] = [128,  64, 128]   # // road          =   7 (8),
    cmap[8] = [244,  35, 232]   # // sidewalk      =   8 (9),
    cmap[9] = [107, 142,  35]   # // vegetation    =   9 (10),
    cmap[10] = [0,   0, 142]    # // car           =  10 (11),
    cmap[11] = [102, 102, 156]  # // wall          =  11 (12),
    cmap[12] = [220, 220,   0]  # // traffic sign  =  12 (13),

    cmap[13] = [60, 250, 240]  # // anomaly       =  13 (14)

    return cmap

class Label2Color(object):
    def __init__(self, cmap):
        self.cmap = cmap

    def __call__(self, lbls):
        return self.cmap[lbls]