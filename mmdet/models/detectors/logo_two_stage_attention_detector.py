import torch
import torch.nn as nn

import os
import sys
import time
import random
from PIL import Image
from xml.etree import ElementTree as ET

from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from mmdet.core import bbox2roi
from mmdet.datasets import build_dataset
from mmcv import Config

@DETECTORS.register_module()
class LogoTwoStageDetectorAttentionDetector(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 support_type=None,
                 support_imgs=None,
                 data_path=None,
                 classes=None,
                 config=None
                ):
        super(LogoTwoStageDetectorAttentionDetector, self).__init__()
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        
        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        """init style and class index"""
        self.style_to_file = {}
        self.file_to_style = {}
        self.file_to_name = {}
        #self.data_path = '/data/zhaozhiyuan/tb_variation/VOCdevkit_all'
        self.data_path = data_path
        self.anno_path = os.path.join(self.data_path, 'VOC2007', 'Annotations')
        self.pic_path = os.path.join(self.data_path, 'VOC2007', 'JPEGImages')
        self.classes = classes
        #self.classes = ['001-CCTV1', '019-fenghuangweishi', '011-history_channel']
        self.config = config
        #self.config = '/home/zhaozhiyuan/workspace/tb_variation/mmdetection-master/configs/faster_rcnn/logo_faster_rcnn_r50_fpn_1x_coco_attention_detector_3classes_normal_init.py'
        self.cfg = Config.fromfile(self.config)
        # no flip for support img
        self.cfg.data.train['pipeline'][3]['flip_ratio'] = 0.0
        self.support_datasets = [build_dataset(self.cfg.data.train)]
        
        self.support_type = support_type
        self.support_imgs = support_imgs
        #self.support_imgs = ['000565', '054646', '058428']
        
        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(LogoTwoStageDetectorAttentionDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_roi_head:
            self.roi_head.init_weights(pretrained)
            
        """
        init style and class index
        """
        for anno in os.listdir(self.anno_path):
            anno_file = ET.parse(os.path.join(self.anno_path, anno))
            name = anno_file.find('object').find('name').text
            style = anno_file.find('object').find('style').text
            self.file_to_style[anno.split('.')[0]] = int(style)
            self.file_to_name[anno.split('.')[0]] = name
            if name not in self.style_to_file:
                self.style_to_file[name] = {}
                self.style_to_file[name][style] = [anno.split('.')[0]]
                continue
            if style not in self.style_to_file[name]:
                self.style_to_file[name][style] = [anno.split('.')[0]]
                continue
            self.style_to_file[name][style].append(anno.split('.')[0])
        
        """
        init class index
        convinient to sample supportss
        """
        self.index_class_style = []
        for i in range(len(self.classes)):
            d = {0:[], 1:[]}
            self.index_class_style.append(d)
        for i in range(self.support_datasets[0].__len__()):
            support = self.support_datasets[0][i]
            img_name = support['img_metas'].data['filename'].split('.')[0].split('/')[-1]
            img_style = self.file_to_style[img_name]
            img_class = int(support['gt_labels'].data[0])
            self.index_class_style[img_class][img_style].append(i)
        
        if self.support_type == 'fixed':
            #self.support_imgs = ['000565', '054646', '058428']
            # sample支持集图片(固定)
            self.support_all = []
            for i, _ in enumerate(self.classes):
                for index in range(self.support_datasets[0].__len__()):
                    support = self.support_datasets[0][index]
                    img_name = support['img_metas'].data['filename'].split('.')[0].split('/')[-1]
                    if img_name == self.support_imgs[i]:
                        self.support_all.append(support)
                        break
        elif self.support_type == 'random':
            # sample支持集图片(不固定)
            self.support_all = []
            for i, _ in enumerate(self.classes):
                style = random.randint(0, 1)
                index = random.choice(self.index_class_style[i][style])
                self.support_all.append(self.support_datasets[0][index])
        
    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        
        """
            building a siamese network
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x
    
    def extract_gt_feat(self, img, gt_bboxes):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        self.roi_head.extract_proposal_feat(img, gt_bboxes)
    
    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      support=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        
        # 提取特征
        img_batch = []
        img_batch.append(img)
        for support in self.support_all:
            s_img = support['img_metas'].data['filename'].split('.')[0].split('/')[-1]
            s_style = self.file_to_style[s_img]
            s_cls = support['gt_labels'].data[0]
            img_batch.append(support['img'].data.unsqueeze(0).to(img.device))
        img_batch = torch.cat(img_batch, dim = 0)
        
        #print(img_batch.shape)
        #sys.exit()
        
        feature_batch = self.extract_feat(img_batch)
    
        # query image feature
        x = [feature[0].unsqueeze(0) for feature in feature_batch]
        # extract support gt bbox feature
        support_gt_feats = []
        for i in range(1, img_batch.shape[0]):
            gt_bboxes_support = self.support_all[i - 1]['gt_bboxes'].data.to(img.device)
            gt_rois_support = bbox2roi([gt_bboxes_support])
            feature_support = [feature[i].unsqueeze(0) for feature in feature_batch]
            gt_bbox_feats_support = self.roi_head.bbox_roi_extractor(feature_support[:self.roi_head.bbox_roi_extractor.num_inputs], gt_rois_support)
            support_gt_feats.append(gt_bbox_feats_support.squeeze(0))
        
        losses = dict()
        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals
    
        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels, support_gt_feats,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        
        losses.update(roi_losses)
        
        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
           
        # 提取特征
        img_batch = []
        img_batch.append(img)
        for support in self.support_all:
            img_batch.append(support['img'].data.unsqueeze(0).to(img.device))
        img_batch = torch.cat(img_batch, dim = 0)
        feature_batch = self.extract_feat(img_batch)
        # query image feature
        x = [feature[0].unsqueeze(0) for feature in feature_batch]
        # extract support gt bbox feature
        support_gt_feats = []
        for i in range(1, img_batch.shape[0]):
            gt_bboxes_support = self.support_all[i - 1]['gt_bboxes'].data.to(img.device)
            gt_rois_support = bbox2roi([gt_bboxes_support])
            feature_support = [feature[i].unsqueeze(0) for feature in feature_batch]
            gt_bbox_feats_support = self.roi_head.bbox_roi_extractor(feature_support[:self.roi_head.bbox_roi_extractor.num_inputs], gt_rois_support)
            support_gt_feats.append(gt_bbox_feats_support.squeeze(0))
        
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, support_gt_feats, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)
