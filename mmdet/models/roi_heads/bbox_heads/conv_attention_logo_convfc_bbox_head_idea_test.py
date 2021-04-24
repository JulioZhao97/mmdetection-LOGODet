#!/usr/bin/env python
# coding: utf-8

import sys
import random
import itertools

import torch
import torch.nn as nn
from torch.autograd import Variable

from mmcv.cnn import ConvModule

from mmdet.models.builder import HEADS
from .attention_logo_bbox_head import AttentionLogoBBoxHead


@HEADS.register_module()
class ConvAttentionLOGOConvFCBBoxHead(AttentionLogoBBoxHead):
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
        super(ConvAttentionLOGOConvFCBBoxHead, self).__init__(*args, **kwargs)
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
                self.global_relation_fc = nn.Linear(self.in_channels * 2, 4)
            else:
                self.global_relation_fc = nn.Linear(self.in_channels * 2, 2)
        self.slice = [
            [(0,7),(0,4)], [(0,7),(3,7)], [(0,4),(0,7)], [(3,7),(0,7)],
            [(0,4),(0,4)], [(0,4),(3,7)], [(3,7),(0,4)], [(3,7),(3,7)],
            [(0,7),(0,7)]
        ]
        self.feature_size = (5,5)
        self.pool = nn.AdaptiveAvgPool2d(self.feature_size)
        self.avgpool = nn.AvgPool2d(kernel_size = 5)
        self.relation_conv = nn.Sequential(
            nn.Conv2d(in_channels = 256 * 2, out_channels = 256 * 4, kernel_size = 1, stride = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 256 * 4, out_channels = 256 * 4, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 256 * 4, out_channels = 256 * 2, kernel_size = 1, stride = 1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size = 5)
        )
        
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
        super(ConvAttentionLOGOConvFCBBoxHead, self).init_weights()
        # conv layers are already initialized by ConvModule
        #for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs]:
        '''
        for module_list in [self.shared_fcs, self.cls_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def get_feature_groups(self, x):
        feature_group = [
            x[:,:,:,:4], x[:,:,:,3:], x[:,:,:4,:], x[:,:,3:,:],
            x[:,:,:4,:4], x[:,:,:4,3:], x[:,:,3:,:4], x[:,:,3:,3:],
            x
        ]
        return [self.pool(feature) for feature in feature_group]
                
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
                _x, _sf = x_global_relation.clone(), sf.clone()
                feature_group_x, feature_group_sf = self.get_feature_groups(_x), self.get_feature_groups(_sf)
                
                weights_all = []
                # cal similarity
                for feature_combination in itertools.product(self.slice, self.slice):
                    # relation conv
                    x_slice = _x[:,:,feature_combination[0][0][0]:feature_combination[0][0][1], feature_combination[0][1][0]:feature_combination[0][1][1]]
                    sf_slice = _sf[:,:,feature_combination[1][0][0]:feature_combination[1][0][1], feature_combination[1][1][0]:feature_combination[1][1][1]]
                    x_part, sf_part = self.pool(x_slice), self.pool(sf_slice)
                    # cal relation
                    x_embedding, sf_embedding = self.avgpool(x_part), self.avgpool(sf_part)
                    x_embedding, sf_embedding = x_embedding.squeeze(3).squeeze(2), sf_embedding.squeeze(3).squeeze(2)
                    similarity = torch.nn.functional.cosine_similarity(x_embedding, sf_embedding, dim = 1)
                    similarity  = torch.mean(similarity, dim=0)
                    weights_all.append(similarity)
                weights_all = torch.nn.functional.softmax(torch.Tensor(weights_all), dim = 0).to(x.device)
                topk = torch.topk(weights_all, k=3, dim=0)
                topk_index, topk_similarity = topk[1], torch.nn.functional.softmax(topk[0], dim = 0)
                
                # relation conv
                cls_logits_all = torch.zeros(x.shape[0], 2).to(x.device)
                for i, feature_combination in enumerate(itertools.product(self.slice, self.slice)):
                    if i not in topk_index:
                        continue
                    x_slice = _x[:,:,feature_combination[0][0][0]:feature_combination[0][0][1], feature_combination[0][1][0]:feature_combination[0][1][1]]
                    sf_slice = _sf[:,:,feature_combination[1][0][0]:feature_combination[1][0][1], feature_combination[1][1][0]:feature_combination[1][1][1]]
                    x_part, sf_part = self.pool(x_slice), self.pool(sf_slice)
                    part_feature = torch.cat([x_part, sf_part], dim = 1)
                    part_feature = self.relation_conv(part_feature).squeeze(3).squeeze(2)
                    cls_logits = self.global_relation_fc(part_feature)
                    index = (topk_index == i).nonzero()[0][0]
                    cls_logits_all += topk_similarity[index] * cls_logits
                
                x_ge = cls_logits_all
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
class ConvAttentionSharedLOGO2FCBBoxHead(ConvAttentionLOGOConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(ConvAttentionSharedLOGO2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

        