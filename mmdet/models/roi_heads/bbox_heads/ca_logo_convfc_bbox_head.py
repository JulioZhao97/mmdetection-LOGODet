#!/usr/bin/env python
# coding: utf-8

import sys
import random

import torch
import torch.nn as nn
from torch.autograd import Variable

from mmcv.cnn import ConvModule

from mmdet.models.builder import HEADS
from .attention_logo_bbox_head import AttentionLogoBBoxHead

import time
import pynvml

@HEADS.register_module()
class CALOGOConvFCBBoxHead(AttentionLogoBBoxHead):
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
        super(CALOGOConvFCBBoxHead, self).__init__(*args, **kwargs)
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
        
        self.relation_conv_x = torch.nn.Conv2d(in_channels = 49, out_channels = 1, kernel_size = 1)
        self.relation_conv_sf = torch.nn.Conv2d(in_channels = 49, out_channels = 1, kernel_size = 1)
        
        '''
        self.avgpool = torch.nn.AvgPool2d(kernel_size = 7)
        self.w = 27
        self.w1 = torch.nn.Parameter(torch.rand(self.roi_feat_area, self.w), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.rand(self.w, self.roi_feat_area), requires_grad=True)
        '''
        
        print(self.score_type)
        print(self.head_config)
        
        # global relation
        if self.head_config[0]:
            self.global_relation_1x1_conv = nn.Conv2d(in_channels = self.in_channels * 2, out_channels = self.in_channels * 2, kernel_size = 1)
            self.global_relation_avgpool = nn.AvgPool2d(kernel_size=7)
            if self.score_type == 'normal':
                self.global_relation_fc = nn.Linear(self.in_channels * 2, 4)
            else:
                self.global_relation_fc = nn.Linear(self.in_channels * 2, 2)
        
        if self.head_config[1]:
        # local relation
            self.local_relation_1x1conv = nn.Conv2d(in_channels = self.in_channels, out_channels = self.in_channels, kernel_size = 1)
            if self.score_type == 'normal':
                self.local_relation_fc = nn.Linear(self.in_channels * 2, 4)
            else:
                self.local_relation_fc = nn.Linear(self.in_channels * 2, 2)
        
        # patch relation
        if self.head_config[2]:
            self.patch_relation_avgpool1 = nn.AvgPool2d(kernel_size=3, stride = 1)
            self.patch_relation_conv1 = nn.Conv2d(in_channels = self.in_channels * 2, out_channels = self.in_channels * 2, kernel_size = 1, stride = 1)
            self.patch_relation_conv2 = nn.Conv2d(in_channels = self.in_channels * 2, out_channels = self.in_channels * 2, kernel_size = 3, stride = 1)
            self.patch_relation_conv3 = nn.Conv2d(in_channels = self.in_channels * 2, out_channels = self.in_channels * 2, kernel_size = 1, stride = 1)
            self.patch_relation_avgpool2 = nn.AvgPool2d(kernel_size=3)
            if self.score_type == 'normal':
                self.patch_relation_fc = nn.Linear(self.in_channels * 2, 4)
            else:
                self.patch_relation_fc = nn.Linear(self.in_channels * 2, 2)
        
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
        super(CALOGOConvFCBBoxHead, self).init_weights()
        # conv layers are already initialized by ConvModule
        #for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs]:
        for module_list in [self.shared_fcs, self.cls_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def attention_module(self, x, sf):
        # attention on x
        x_1 = x.clone()
        x_1 = x_1.permute(0,2,3,1)
        x_1 = x_1.reshape(x_1.shape[0], x_1.shape[1] * x_1.shape[2], x_1.shape[3])
        sf_1 = sf.reshape(sf.shape[0], sf.shape[1] * sf.shape[2])
        
        # normalize
        x_1_n = torch.nn.functional.normalize(x_1, p=2, dim=1)
        sf_1_n = torch.nn.functional.normalize(sf_1, p=2, dim=0)
        correlation_map = torch.matmul(x_1_n, sf_1_n)
        correlation_map_x = correlation_map.reshape(correlation_map.shape[0], correlation_map.shape[1], 7, 7)
        
        # attention on x
        attntion_map_x = torch.sigmoid(self.relation_conv_x(correlation_map_x))
        x_1 = x_1.reshape(x_1.shape[0], 7, 7, x_1.shape[2])
        x_1 = x_1.permute(0,3,1,2)
        att_x = x_1 * attntion_map_x
        
        '''
        cor_all = []
        for i in range(correlation_map_x.shape[0]):
            cur_cor = correlation_map_x[i].unsqueeze(0)
            m = self.avgpool(cur_cor).squeeze(3).squeeze(2)
            m = torch.mm(m, self.w1)
            m = torch.mm(m, self.w2)
            m = m.unsqueeze(2).unsqueeze(3)
            cor = torch.nn.functional.conv2d(cur_cor, m)
            cor_all.append(cor)
        cor_all = torch.cat(cor_all, dim = 0)
        # reshape to normal
        x_1 = x_1.reshape(x_1.shape[0], 7, 7, x_1.shape[2])
        x_1 = x_1.permute(0,3,1,2)
        att_x = x_1 * cor_all + x_1
        '''
        
        # attention on sf
        correlation_map_sf = correlation_map.permute(0,2,1).reshape(correlation_map.shape[0], correlation_map.shape[1], 7, 7)
        attntion_map_sf = torch.sigmoid(self.relation_conv_sf(correlation_map_sf))
        # reshape to normal
        sf_2 = sf.clone().unsqueeze(0).repeat(x.shape[0],1,1,1)
        att_sf = sf_2 * attntion_map_sf
        
        return att_x, att_sf
                    
    def forward(self, x, support_feature):
        
        x_global_relation = x
        x_local_relation = x
        x_patch_relation = x
        
        cls_score = torch.zeros(x.shape[0], 4).to(x.device)
        
        # global relation module
        if self.head_config[0]:
            global_relation_score = []
            bg_scores = []
            for i, sf in enumerate(support_feature):
                # attention module
                _x_global_relation, _sf = self.attention_module(x_global_relation, sf)
                #sf = sf.repeat(x_global_relation.shape[0], 1, 1, 1)
                x_ge = torch.cat([_x_global_relation, _sf], dim = 1)
                x_ge = self.relu(self.global_relation_1x1_conv(x_ge))
                x_ge = self.global_relation_avgpool(x_ge)
                x_ge = x_ge.squeeze(3).squeeze(2)
                x_ge = self.global_relation_fc(x_ge)
                if self.score_type == 'normal':
                    global_relation_score.append(x_ge)
                else:
                    global_relation_score.append(x_ge[:, 0].unsqueeze(1))
                    bg_scores.append(x_ge[:, 1].unsqueeze(1))
            # different score method
            if self.score_type == 'normal':
                global_relation_score = (global_relation_score[0]+global_relation_score[1]+global_relation_score[2]) / 3
            elif self.score_type == 'mean':
                bg_score = torch.mean(torch.cat(bg_scores, dim = 1), dim = 1).unsqueeze(1)
                global_relation_score.append(bg_score)
                global_relation_score = torch.cat(global_relation_score, dim = 1)
            elif self.score_type == 'max':
                bg_score = torch.max(torch.cat(bg_scores, dim = 1), dim = 1)[0].unsqueeze(1)
                global_relation_score.append(bg_score)
                global_relation_score = torch.cat(global_relation_score, dim = 1)
            cls_score += global_relation_score
        
        return cls_score


@HEADS.register_module()
class CASharedLOGO2FCBBoxHead(CALOGOConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(CASharedLOGO2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

        