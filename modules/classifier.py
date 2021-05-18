import torch.nn as nn
import torch.nn.functional as F


def get_classifier(classifier_type):
    if classifier_type == 'standard':
        return Classifier
    elif classifier_type == 'cosine':
        return CosineClassifier


class Classifier(nn.Module):

    def __init__(self, channels, classes, deepsup=False):
        super(Classifier, self).__init__()
        self.channels = channels
        self.classes = classes
        self.cls = nn.Conv2d(channels, classes, 1)

        self.deepsup = deepsup
        if deepsup:
            self.cls_deepsup = nn.Conv2d(channels, classes, 1, 1, 0)

    def forward(self, x, x_deepsup):
        logits_deepsup = self.cls_deepsup(x_deepsup) if self.deepsup else None
        logits = self.cls(x)

        return logits, logits_deepsup


class CosineClassifier(nn.Module):
    def __init__(self, channels, classes, deepsup=False):
        super().__init__()
        self.channels = channels
        self.scaler = 10.
        self.classes = classes
        self.tot_classes = 0

        self.cls = nn.Conv2d(channels, classes, 1, bias=False)

        self.deepsup = deepsup

        if deepsup:
            self.cls_deepsup = nn.Conv2d(channels, classes, 1, 1, 0, bias=False)

    def forward(self, x, x_deepsup):
        out = F.normalize(x, p=2, dim=1)
        logits = self.scaler * F.conv2d(out, F.normalize(self.cls.weight, dim=1, p=2))

        if self.deepsup:
            out = F.normalize(x_deepsup, p=2, dim=1)
            logits_deepsup = self.scaler * F.conv2d(out, F.normalize(self.cls_deepsup.weight, dim=1, p=2))
        else:
            logits_deepsup = None

        return logits, logits_deepsup