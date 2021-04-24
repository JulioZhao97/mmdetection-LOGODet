import sys
import torch
import torch.nn as nn
import logging
from ..builder import BACKBONES
from mmcv.cnn import ConvModule, constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

@BACKBONES.register_module()
class YoloLiteLogo(nn.Module):
    def __init__(self):
        super(YoloLiteLogo, self).__init__()
        self.c1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, stride = 1, padding = 1)
        self.c2 = nn.Conv2d(in_channels = 16, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.c3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
        self.c4 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)
        self.c5 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)
        self.mp = nn.MaxPool2d(kernel_size = 2)
        self.relu = nn.ReLU(inplace = False)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        outs = []
        x = self.mp(self.relu(self.c1(x))) # 400
        x = self.mp(self.relu(self.c2(x))) # 200
        outs.append(x)
        x = self.mp(self.relu(self.c3(x))) # 100
        outs.append(x)
        x = self.mp(self.relu(self.c4(x))) # 50
        outs.append(x)
        x = self.mp(self.relu(self.c5(x))) # 25
        outs.append(x)
        
        return outs
