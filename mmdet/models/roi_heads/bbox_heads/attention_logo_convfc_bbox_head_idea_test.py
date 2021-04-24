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


@HEADS.register_module()
class IdeaTestAttentionLOGOConvFCBBoxHead(AttentionLogoBBoxHead):
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
        super(IdeaTestAttentionLOGOConvFCBBoxHead, self).__init__(*args, **kwargs)
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
        #self.shared_convs, self.shared_fcs, last_layer_dim = \
        #    self._add_conv_fc_branch(
        #        self.num_shared_convs, self.num_shared_fcs, self.in_channels,
        #        True)
        #self.shared_out_channels = last_layer_dim

        # add cls specific branch
        #self.cls_convs, self.cls_fcs, self.cls_last_dim = \
        #    self._add_conv_fc_branch(
        #        self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)
        
        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        #if self.with_cls:
        #    self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes + 1)
        
        print(self.score_type)
        print(self.head_config)
        
        # global relation
        if self.head_config[0]:
            self.global_relation_1x1_conv = nn.Conv2d(in_channels = self.in_channels * 2, out_channels = self.in_channels * 2, kernel_size = 1)
            self.global_relation_avgpool = nn.AvgPool2d(kernel_size=7)
            if self.score_type == 'normal':
                self.global_relation_fc = nn.Sequential(
                    nn.Linear(512 * 32, 4096),
                    nn.Dropout(p = 0.5),
                    nn.Linear(4096, 4)
                )
            else:
                self.global_relation_fc = nn.Sequential(
                    nn.Linear(512 * 32, 4096),
                    nn.Dropout(p = 0.5),
                    nn.Linear(4096, 2)
                )
        self.avgpool1 = torch.nn.AvgPool2d(kernel_size = (7,3), stride = (1,2))
        self.avgpool2 = torch.nn.AvgPool2d(kernel_size = (3,7), stride = (2,1))
        self.avgpool3 = torch.nn.AvgPool2d(kernel_size = (3,3), stride = (2,2))
        self.avgpool4 = torch.nn.AvgPool2d(kernel_size = (7,7))
        self.meta_x_6 = torch.nn.Sequential(
            nn.Linear(in_features = 16, out_features = 16),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(in_features = 16, out_features = 6),
            nn.ReLU(inplace=True),
            nn.Linear(in_features = 6, out_features = 16),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(in_features = 16, out_features = 16),
        )
        self.meta_x_12 = torch.nn.Sequential(
            nn.Linear(in_features = 16, out_features = 16),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(in_features = 16, out_features = 12),
            nn.ReLU(inplace=True),
            nn.Linear(in_features = 12, out_features = 16),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(in_features = 16, out_features = 16),
        )
        self.meta_x = [self.meta_x_6, self.meta_x_12]
        self.meta_sf_6 = torch.nn.Sequential(
            nn.Linear(in_features = 16, out_features = 16),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(in_features = 16, out_features = 6),
            nn.ReLU(inplace=True),
            nn.Linear(in_features = 6, out_features = 16),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(in_features = 16, out_features = 16),
        )
        self.meta_sf_12 = torch.nn.Sequential(
            nn.Linear(in_features = 16, out_features = 16),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(in_features = 16, out_features = 12),
            nn.ReLU(inplace=True),
            nn.Linear(in_features = 12, out_features = 16),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(in_features = 16, out_features = 16),
        )
        
        self.meta_sf = [self.meta_sf_6, self.meta_sf_12]
        
        self.topk = 10
        
        if self.head_config[1]:
        # local relation
            self.local_relation_1x1conv = nn.Conv2d(in_channels = self.in_channels, out_channels = self.in_channels, kernel_size = 1)
            if self.score_type == 'normal':
                self.local_relation_fc = nn.Sequential(
                    nn.Linear(512 * 32, 2048),
                    nn.Dropout(p = 0.5),
                    nn.Linear(2048, 4)
                )
            else:
                self.local_relation_fc = nn.Sequential(
                    nn.Linear(512 * 32, 2048),
                    nn.Dropout(p = 0.5),
                    nn.Linear(2048, 2)
                )
        
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
        super(IdeaTestAttentionLOGOConvFCBBoxHead, self).init_weights()
        # conv layers are already initialized by ConvModule
        #for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs]:
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias, 0)

    def get_feature_group(self, x):
        x1 = self.avgpool1(x).squeeze(2)
        x2 = self.avgpool2(x).squeeze(3)
        x3 = self.avgpool3(x)
        x3 = x3.reshape(x3.shape[0], x3.shape[1], x3.shape[2] * x3.shape[3])
        x4 = self.avgpool4(x).squeeze(3)
        feature_group = torch.cat([x1,x2,x3,x4], dim = 2)
        return feature_group
                
    def attention_module(self, x, sf):
        _x, _sf = x.clone(), sf.clone()
        feature_group_x, feature_group_sf = self.get_feature_group(x), self.get_feature_group(sf)
        
        # correlation
        correlation_x_sf = torch.bmm(
            nn.functional.normalize(feature_group_x.permute(0,2,1),p=2,dim=2), 
            nn.functional.normalize(feature_group_sf,p=2,dim=1)
        )
        '''
        if self.correlation_type == 'softmax':
            print('ohuo1')
            correlation_x_sf = torch.softmax(correlation_x_sf, dim = 2)
        elif self.correlation_type == 'topk':
            print('ohuo2')
            # using topk
            thresholds = torch.topk(correlation_x_sf, k=self.topk, dim=2)[0]
            thresholds = torch.min(thresholds, dim = 2)[0].unsqueeze(2)
            correlation_x_sf = correlation_x_sf * (correlation_x_sf > thresholds).long()
        '''
        correlation_sf_x = torch.bmm(
            nn.functional.normalize(feature_group_sf.permute(0,2,1),p=2,dim=2), 
            nn.functional.normalize(feature_group_x,p=2,dim=1)
        )
        '''
        if self.correlation_type == 'softmax':
            correlation_sf_x = torch.softmax(correlation_sf_x, dim = 2)
        elif self.correlation_type == 'topk':
            # using topk
            thresholds = torch.topk(correlation_sf_x, k=self.topk, dim=2)[0]
            thresholds = torch.min(thresholds, dim = 2)[0].unsqueeze(2)
            correlation_sf_x = correlation_sf_x * (correlation_sf_x > thresholds).long()
        '''
        
        
        # meta-learning for x
        select_x_sf = [meta(correlation_x_sf) for meta in self.meta_x]
        feature_group_x_selected = [torch.bmm(feature_group_x, s_x_sf) for s_x_sf in select_x_sf]
        feature_group_x_selected = torch.cat(feature_group_x_selected, dim = 2)
        # meta-learning for sf
        select_sf_x = [meta(correlation_sf_x) for meta in self.meta_sf]
        feature_group_sf_selected = [torch.bmm(feature_group_sf, s_sf_x) for s_sf_x in select_sf_x]
        feature_group_sf_selected = torch.cat(feature_group_sf_selected, dim = 2)
        return feature_group_x_selected, feature_group_sf_selected
        
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
                sf = sf.repeat(x_global_relation.shape[0], 1, 1, 1)
                feature_group_x, feature_group_sf = self.attention_module(x_global_relation, sf)
                x_ge = torch.cat([feature_group_x, feature_group_sf], dim = 1)
                x_ge = torch.flatten(x_ge, 1, 2)
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
        
        
        
        # local relation module
        if self.head_config[1]:
            local_relation_score = []
            bg_scores = []
            for i, sf in enumerate(support_feature):
                _x_lr = self.relu(self.local_relation_1x1conv(x_local_relation))
                sf = sf.unsqueeze(0)
                _sf = self.relu(self.local_relation_1x1conv(sf))
                x_lr = self.relu(torch.nn.functional.conv2d(_x_lr, _sf.permute(1,0,2,3), groups = 256))
                x_lr = x_lr.squeeze(3).squeeze(2)
                x_lr = self.local_relation_fc(x_lr)
                if self.score_type == 'normal':
                    local_relation_score.append(x_lr)
                else:
                    local_relation_score.append(x_lr[:, 0].unsqueeze(1))
                    bg_scores.append(x_lr[:, 1].unsqueeze(1))
            # different score method
            if self.score_type == 'normal':
                local_relation_score = (local_relation_score[0]+local_relation_score[1]+local_relation_score[2]) / 3
            elif self.score_type == 'mean':
                bg_score = torch.mean(torch.cat(bg_scores, dim = 1), dim = 1).unsqueeze(1)
                local_relation_score.append(bg_score)
                local_relation_score = torch.cat(local_relation_score, dim = 1)
            elif self.score_type == 'max':
                bg_score = torch.max(torch.cat(bg_scores, dim = 1), dim = 1)[0].unsqueeze(1)
                local_relation_score.append(bg_score)
                local_relation_score = torch.cat(local_relation_score, dim = 1)
            cls_score += local_relation_score
        
        # patch relation module
        if self.head_config[2]:
            patch_relation_score = []
            bg_scores = []
            for i, sf in enumerate(support_feature):
                sf = sf.repeat(x_patch_relation.shape[0], 1, 1, 1)
                x_pr = torch.cat([x_patch_relation, sf], dim = 1)
                x_pr = self.patch_relation_avgpool1(x_pr)
                x_pr = self.relu(self.patch_relation_conv1(x_pr))
                x_pr = self.relu(self.patch_relation_conv2(x_pr))
                x_pr = self.relu(self.patch_relation_conv3(x_pr))
                x_pr = self.patch_relation_avgpool2(x_pr)
                x_pr = x_pr.squeeze(3).squeeze(2)
                x_pr = self.patch_relation_fc(x_pr)
                if self.score_type == 'normal':
                    patch_relation_score.append(x_pr)
                else:
                    patch_relation_score.append(x_pr[:, 0].unsqueeze(1))
                    bg_scores.append(x_pr[:, 1].unsqueeze(1))
            # different score method
            if self.score_type == 'normal':
                patch_relation_score = (patch_relation_score[0]+patch_relation_score[1]+patch_relation_score[2]) / 3
            elif self.score_type == 'mean':
                bg_score = torch.mean(torch.cat(bg_scores, dim = 1), dim = 1).unsqueeze(1)
                patch_relation_score.append(bg_score)
                patch_relation_score = torch.cat(patch_relation_score, dim = 1)
            elif self.score_type == 'max':
                bg_score = torch.max(torch.cat(bg_scores, dim = 1), dim = 1)[0].unsqueeze(1)
                patch_relation_score.append(bg_score)
                patch_relation_score = torch.cat(patch_relation_score, dim = 1)
            cls_score += patch_relation_score
        
        cls_score = cls_score / (torch.sum((torch.Tensor(self.head_config)==True).int()))
        
        return cls_score


@HEADS.register_module()
class IdeaTestAttentionSharedLOGO2FCBBoxHead(IdeaTestAttentionLOGOConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(IdeaTestAttentionSharedLOGO2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

        