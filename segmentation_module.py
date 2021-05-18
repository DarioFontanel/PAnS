import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as functional

import inplace_abn
from inplace_abn import InPlaceABNSync, InPlaceABN, ABN

from functools import partial

import models
from modules import PyramidPoolingModule, get_classifier


def make_model(opts):
    if opts.norm_act == 'iabn_sync':
        norm = partial(InPlaceABNSync, activation="leaky_relu", activation_param=.01)
    elif opts.norm_act == 'iabn':
        norm = partial(InPlaceABN, activation="leaky_relu", activation_param=.01)
    elif opts.norm_act == 'abn':
        norm = partial(ABN, activation="leaky_relu", activation_param=.01)
    else:
        norm = nn.BatchNorm2d  # not synchronized, can be enabled with apex

    # === BODY
    body = models.__dict__[f'net_{opts.backbone}'](norm_act=norm, output_stride=opts.output_stride)
    if not opts.no_pretrained:
        pretrained_path = f'pretrained/{opts.backbone}_{opts.norm_act}.pth.tar'
        pre_dict = torch.load(pretrained_path, map_location='cpu')
        del pre_dict['state_dict']['classifier.fc.weight']
        del pre_dict['state_dict']['classifier.fc.bias']

        body.load_state_dict(pre_dict['state_dict'])
        del pre_dict  # free memory

    # === HEAD
    if opts.head == 'PPM':
        deepsup = True
        head_channels = 512
        head = PyramidPoolingModule(body.out_channels, head_channels, norm_act=norm)

    else:
        raise NotImplementedError(f"Head _{opts.head}_ not implemented.")

    # === CLASSIFIER
    classifier = get_classifier(opts.classifier)
    cls = classifier(head.out_channels, opts.num_classes, deepsup)

    model = SegmentationModule(body, head, cls, opts.fix_bn)

    return model


class SegmentationModule(nn.Module):
    def __init__(self, body, head, cls, fix_bn=False):
        super(SegmentationModule, self).__init__()
        self.body = body
        self.head = head
        self.cls = cls

        if fix_bn:
            self.fix_bn()

    def _network(self, x, body_and_head=False):

        x_b_all, x_b_last = self.body(x)
        x_h = self.head((x_b_all, x_b_last))
        if body_and_head:
            x_h.update({'body': x_b_last})
            return x_h
        return x_h

    def freeze_network(self):
        for par in self.body.parameters():
            par.requires_grad = False
        for par in self.head.parameters():
            par.requires_grad = False
        for par in self.cls.parameters():
            par.requires_grad = False

    def forward(self, x, body_and_head=False, custom_outsize=None, interpolate=True):
        out = self._network(x, body_and_head)
        out_size = x.shape[-2:]
        features, features_deepsup = out['outputs'], out['outputs_deepsup']
        if body_and_head:
            feat_body = out['body']
            # body_output, _ = self.intermediate_cls(feat_body, None)

        # SEGMENTATION TASK
        sem_output, sem_output_deepsup = self.cls(features, features_deepsup)

        # === PREDICTIONS
        if interpolate:
            sem_output = functional.interpolate(sem_output, size=torch.Size(custom_outsize) if custom_outsize
                                                    is not None else out_size, mode="bilinear", align_corners=False)

            if sem_output_deepsup is not None:
                sem_output_deepsup = functional.interpolate(sem_output_deepsup, size=out_size, mode="bilinear",  align_corners=False)

        if body_and_head:
            return sem_output, sem_output_deepsup, features, feat_body

        return sem_output, sem_output_deepsup, features

    def fix_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, inplace_abn.ABN):
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def freeze(self):
        for p in self.body.parameters():
            p.requires_grad = False
        for p in self.head.parameters():
            p.requires_grad = False
        for p in self.cls.parameters():
            p.requires_grad = False
