import torch
import torch.nn as nn


# pyramid pooling, deep supervision
class PyramidPoolingModule(nn.Module):
    def __init__(self,
                 in_channels=2048,
                 out_channels=512,
                 norm_act=nn.BatchNorm2d,
                 pool_scales=(1, 2, 3, 6)):

        super(PyramidPoolingModule, self).__init__()

        self.out_channels = out_channels
        self.ppm = []

        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(in_channels, 512, kernel_size=1, bias=False),
                norm_act(512),
            ))
        self.ppm = nn.ModuleList(self.ppm)
        self.conv_last = nn.Sequential(
            nn.Conv2d(in_channels + len(pool_scales) * 512, 512,
                      kernel_size=3, padding=1, bias=False),
            norm_act(self.out_channels),
            nn.Dropout2d(0.1),

        )
        self.cbr_deepsup = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, stride=1, padding=1, bias=False),
            norm_act(in_channels // 4)
        )
        self.dropout_deepsup = nn.Dropout2d(0.1)

    def forward(self, conv_out):
        all_conv_out, _ = conv_out
        conv5 = all_conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        outputs = self.conv_last(ppm_out)

        # deep sup
        conv4 = all_conv_out[-2]
        outputs_deepsup = self.cbr_deepsup(conv4)
        outputs_deepsup = self.dropout_deepsup(outputs_deepsup)

        return {'outputs': outputs,
                'outputs_deepsup': outputs_deepsup}