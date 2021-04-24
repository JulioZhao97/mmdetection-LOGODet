#!/usr/bin/env python
# coding: utf-8

import sys
import random

import torch
import torch.nn as nn
from torch.autograd import Variable

from mmcv.cnn import ConvModule

from mmdet.models.builder import HEADS
from .logo_bbox_head import LogoBBoxHead


@HEADS.register_module()
class LOGOConvFCREBBoxHead(LogoBBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 #num_reg_convs=0,
                 #num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 *args,
                 **kwargs):
        super(LOGOConvFCREBBoxHead, self).__init__(*args, **kwargs)
        #assert (num_shared_convs + num_shared_fcs + num_cls_convs + num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        #if num_cls_convs > 0 or num_reg_convs > 0:
        #    assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)
        
        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes + 1)
        
        self.attention_conv = nn.Conv2d(in_channels = 19, out_channels = 1, kernel_size = 3, stride = 1, padding = 1)
        
    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        super(LOGOConvFCREBBoxHead, self).init_weights()
        # conv layers are already initialized by ConvModule
        #for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs]:
        for module_list in [self.shared_fcs, self.cls_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, labels = None):
        if self.training:
            for (i, label) in enumerate(labels):
                if label != 19:
                    # random erasing
                    p = random.random()
                    if p > 0.5:
                        direct = random.random()
                        if direct > 0.5:
                            #抹除下面
                            r = random.randint(3,4)
                            x[i, :, r:, :] = 0
                        else:
                            r = random.randint(3,4)
                            x[i, :, :, r:] = 0
                    '''
                    # random channel shuffle
                    p = random.random()
                    if p > 0.5:
                        r = random.choice([2,4,8,16,32,64,128])
                        proposal = x[i]
                        proposal = proposal.reshape(r, proposal.shape[0] // r, proposal.shape[1], proposal.shape[2])
                        proposal = proposal.permute(1,0,2,3)
                        x[i] = proposal.reshape(proposal.shape[0] * proposal.shape[1], proposal.shape[2], proposal.shape[3])
                    '''
                    '''
                    p = random.random()
                    if p > 0.5:
                        r = torch.rand(7, 7)
                        x[i] = x[i] * r.to(x.device)
                    '''
        
        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.flatten(1)
            for fc in self.shared_fcs:
                x= self.relu(fc(x))
                
        x_cls = x
        # separate branches
        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        
        return cls_score, None, None


@HEADS.register_module()
class SharedLOGO2FCBBoxREHead(LOGOConvFCREBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(SharedLOGO2FCBBoxREHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

        