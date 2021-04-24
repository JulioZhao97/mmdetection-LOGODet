from ..builder import DETECTORS
from .logo_two_stage_attention_detector import LogoTwoStageDetectorAttentionDetector


@DETECTORS.register_module()
class LogoFasterRCNNAttentionDetector(LogoTwoStageDetectorAttentionDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 support_type=None):
        super(LogoFasterRCNNAttentionDetector, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            support_type=support_type)
