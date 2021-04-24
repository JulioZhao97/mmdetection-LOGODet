from mmdet.core import eval_map, eval_recalls
from .builder import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module()
class VOCDataset(XMLDataset):

    '''
    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')
    '''

    '''
    CLASSES = ('001-CCTV1', '002-beijingweishi', '003-tianjinweishi', '004-jiangsuweishi', '005-shandongweishi', '006-henanweishi', '007-guangdongweishi', '008-shenzhenweishi', '009-lvyouweishi', '010-sichuanweishi', '011-history_channel', '012-ABC', '013-FOX', '014-VOA', '015-DWTV', '016-CBS', '017-iTV', '018-NHK', '019-fenghuangweishi', '020-tbs')
    '''

    '''    
    CLASSES = ('001-CCTV1','002-beijingweishi','003-tianjinweishi','004-CNN','005-shandongweishi','006-henanweishi','007-guangdongweishi','008-shenzhenweishi','010-sichuanweishi','011-history_channel','014-VOA','015-DWTV','016-CBS','017-KTN','017-iTV','018-NOWTHIS','019-fenghuangweishi','019-TVBS','020-tbs')
    '''
    '''
    CLASSES = ('zhongguoneimu', 'ziyouzhongguo', 'SVT', 'ABC_australia_news', 'xiwangzhisheng', 'gongshi', 'aboluoxinwen', 'jiaguozhisheng', 'shijiemenxinwenwang', 'falungong', 'senzheshentan', 'sanlidianshitai', 'zhongguoreping', 'baoshengfangtan', 'zhongguoguangbogongsi', 'redianhudong', 'xiaominzhixin', 'zhonghuadianshigongsi', 'RFI', 'xinwenshishibao', 'wanweiduzhe', 'BBC_news', 'dajiyuan', 'zhenlizhiguang', 'zhoumochaguan', 'huanqiuzhiji', 'ABC_news', 'zhongtianxinwen', 'FT_zhongwenwang', 'minshi', 'fanqiangbikan', 'mingjinghuopai', 'wenqiandeshijiezhoubao', 'ZDF', 'CNN_news', 'bowenshe', 'quanqiushiye', 'ziyouyazhou', 'VOA', 'DW_TV')
    '''

    CLASSES = ('zhongguoneimu', 'xiwangzhisheng', 'FOX', 'zhongguoguangbogongsi', 'shijiemenxinwenwang', 'CNN_news', 'falungong', 'wanweiduzhe', 'zhonghuadianshigongsi', 'ZDF', 'jiaguozhisheng', 'RFI', 'senzheshentan', 'dajiyuan', 'baoshengfangtan', 'redianhudong', 'mingjinghuopai', 'ziyouzhongguo', 'BBC_news', 'xinwenshishibao', 'zhongtianxinwen', 'sanlidianshitai', 'DW_TV', 'zhoumochaguan', 'bowenshe', 'huanqiuzhiji', 'minshi', 'zhongguojinwen', 'zhongguoreping', 'FT_zhongwenwang', 'fanqiangbikan', 'VOA', 'gongshi', 'SVT', 'aboluoxinwen', 'wenqiandeshijiezhoubao', 'ABC_news', 'quanqiushiye', 'ABC_australia_news', 'ziyouyazhou', 'zhenlizhiguang')


    def __init__(self, **kwargs):
        super(VOCDataset, self).__init__(**kwargs)
        if 'VOC2007' in self.img_prefix:
            self.year = 2007
        elif 'VOC2012' in self.img_prefix:
            self.year = 2012
        else:
            raise ValueError('Cannot infer dataset year from img_prefix')

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        """Evaluate in VOC protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'mAP', 'recall'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple], optional): Scale ranges for evaluating
                mAP. If not specified, all bounding boxes would be included in
                evaluation. Default: None.

        Returns:
            dict[str, float]: AP/recall metrics.
        """
   
        print(iou_thr)

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = {}
        if metric == 'mAP':
            assert isinstance(iou_thr, float)
            if self.year == 2007:
                ds_name = self.CLASSES
            else:
                ds_name = self.CLASSES
            mean_ap, _ = eval_map(
                results,
                annotations,
                scale_ranges=None,
                iou_thr=iou_thr,
                dataset=ds_name,
                logger=logger)
            eval_results['mAP'] = mean_ap
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            if isinstance(iou_thr, float):
                iou_thr = [iou_thr]
            recalls = eval_recalls(
                gt_bboxes, results, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thr):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results
