CUDA_VISIBLE_DEVICES=2 python tools/test.py configs/ssd/ssd300_coco.py work_dirs/ssd300_coco/epoch_1.pth --eval mAP
CUDA_VISIBLE_DEVICES=2 python tools/test.py configs/ssd/ssd300_coco.py work_dirs/ssd300_coco/epoch_2.pth --eval mAP
CUDA_VISIBLE_DEVICES=2 python tools/test.py configs/ssd/ssd300_coco.py work_dirs/ssd300_coco/epoch_3.pth --eval mAP
CUDA_VISIBLE_DEVICES=2 python tools/test.py configs/ssd/ssd300_coco.py work_dirs/ssd300_coco/epoch_4.pth --eval mAP
CUDA_VISIBLE_DEVICES=2 python tools/test.py configs/ssd/ssd300_coco.py work_dirs/ssd300_coco/epoch_5.pth --eval mAP
CUDA_VISIBLE_DEVICES=2 python tools/test.py configs/ssd/ssd300_coco.py work_dirs/ssd300_coco/epoch_6.pth --eval mAP