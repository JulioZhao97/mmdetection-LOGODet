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
class CCALOGOConvFCBBoxHead(AttentionLogoBBoxHead):
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
        super(CCALOGOConvFCBBoxHead, self).__init__(*args, **kwargs)
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
        
        cascade_level = 2
        self.topk = [32, 16, 8]
        self.avgpool_all = [nn.AvgPool2d(kernel_size = (i+1)*2 + 1, stride = 1) for i in range(cascade_level)]
        self.fc1 = nn.Linear(256 * (7 * 7 + sum(self.topk)), 4096)
        self.fc2 = nn.Linear(4096, 2)
        
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
        super(CCALOGOConvFCBBoxHead, self).init_weights()
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

    def cascade_avg_pooling(self, x):
        pooled_features = []
        pooled_features.append(x)
        for avg in self.avgpool_all:
            pooled_features.append(avg(x))
        return pooled_features
                
    def cca_module(self, x, sf):
        _x, _sf = x.clone(), sf.clone()
        find_features = []
        
        '''
        for i in range(_x.shape[0]):
            cur_x, cur_sf = _x[i], sf[i]
            pool_x, pool_sf = self.cascade_avg_pooling(cur_x), self.cascade_avg_pooling(cur_sf)
            top_k_features_all = []
            for i, pooled_x in enumerate(pool_x):
                pooled_x_n = torch.nn.functional.normalize(pooled_x, p=2, dim=0)
                correlation_all = []
                w_x, h_x = pooled_x_n.shape[1], pooled_x_n.shape[2]
                for pooled_sf in pool_sf:
                    w_sf, h_sf = pooled_sf.shape[1], pooled_sf.shape[2]
                    pooled_sf_n = torch.nn.functional.normalize(pooled_sf, p=2, dim=0)
                    _pooled_x_n = pooled_x_n.permute(1,2,0).reshape(w_x * h_x, 256)
                    _pooled_sf_n = pooled_sf_n.reshape(256, w_sf * h_sf)
                    correlation = torch.mm(_pooled_x_n, _pooled_sf_n)
                    correlation = torch.mean(correlation, dim=1)
                    correlation_all.append(correlation.unsqueeze(1))
                correlation_all = torch.cat(correlation_all, dim=1)
                correlation_all = torch.mean(correlation_all, dim=1)
                top_k_index = torch.topk(correlation_all, self.topk[i])[1]
                top_k_features = pooled_x.reshape(256, w_x * h_x).permute(1, 0)[top_k_index]
                top_k_features_all.append(top_k_features)
            top_k_features_all = torch.cat(top_k_features_all, dim = 0)
            top_k_features_all = torch.flatten(top_k_features_all, 0, 1).unsqueeze(0)
            find_features.append(top_k_features_all)
        find_features = torch.cat(find_features, 1)
        '''
        
        
        find_features = []
        pool_x, pool_sf = self.cascade_avg_pooling(_x), self.cascade_avg_pooling(_sf)
        top_k_features_all = []
        for i, pooled_x in enumerate(pool_x):
            pooled_x_n = torch.nn.functional.normalize(pooled_x, p=2, dim=1)
            correlation_all = []
            w_x, h_x = pooled_x_n.shape[2], pooled_x_n.shape[3]
            for pooled_sf in pool_sf:
                w_sf, h_sf = pooled_sf.shape[1], pooled_sf.shape[2]
                pooled_sf_n = torch.nn.functional.normalize(pooled_sf, p=2, dim=0)
                _pooled_sf_n = pooled_sf_n.reshape(256, w_sf * h_sf)
                
                _pooled_x_n = pooled_x_n.permute(0,2,3,1).reshape(x.shape[0], w_x * h_x, 256).reshape(x.shape[0]*w_x*h_x, 256)
                correlation = torch.matmul(_pooled_x_n, _pooled_sf_n).reshape(x.shape[0], w_x * h_x, w_sf * h_sf)
                correlation = torch.mean(correlation, dim=2)
                correlation_all.append(correlation.unsqueeze(2))
            correlation_all = torch.cat(correlation_all, dim=2)
            correlation_all = torch.mean(correlation_all, dim=2)
            #correlation_all.register_hook(lambda grad: grad.contiguous())
            
            
            _correlation_all = correlation_all.clone()
            top_k_index = torch.topk(_correlation_all,k=self.topk[i],dim=1)[1]
            _pooled_x = pooled_x.permute(0,2,3,1).reshape(x.shape[0],w_x * h_x,256)
            if self.training:
                pooled_x.register_hook(lambda grad: grad.contiguous())
            top_k_features = [_pooled_x[i][top_k_index[i].long()].unsqueeze(0) for i in range(_pooled_x.shape[0])]
            top_k_features = torch.cat(top_k_features, dim=0)
            if self.training:
                top_k_features.register_hook(lambda grad: grad.contiguous())
            top_k_features_all.append(top_k_features)
            
            
            '''
            pooled_x.register_hook(lambda grad: grad.contiguous())
            _pooled_x = pooled_x.permute(0,2,3,1).reshape(x.shape[0],w_x * h_x,256)
            top_k_features = []
            for j in range(correlation_all.shape[0]):
                cor = correlation_all[j]
                top_k_index = torch.topk(cor,k=self.topk[i],dim=0)[1]
                top_k_features.append(_pooled_x[j][top_k_index].unsqueeze(0))
            top_k_features = torch.cat(top_k_features, dim = 0)
            top_k_features.register_hook(lambda grad: grad.contiguous())
            top_k_features_all.append(top_k_features)
            '''
        top_k_features_all = torch.cat(top_k_features_all, dim = 1)
        find_features = torch.flatten(top_k_features_all, 1, 2)
        
        return find_features
            
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
                find_features = self.cca_module(x_global_relation, sf)
                sf = torch.flatten(sf.unsqueeze(0).repeat(find_features.shape[0], 1, 1, 1), 1, 3)
                x_ge = torch.cat([find_features, sf], dim = 1)
                x_ge = self.fc2(self.relu(self.fc1(x_ge)))
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
class CCASharedLOGO2FCBBoxHead(CCALOGOConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(CCASharedLOGO2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

        