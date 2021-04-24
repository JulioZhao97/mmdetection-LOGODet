# mmdetection-LOGODet
## Introduction
This is a LOGO detection project based on mmdetection framework

## Environment
cuda, cudatoolkit = 10.0

torch = 1.4.0

torchvision = 0.5.0

mmdetetcion = 2.6.0

mmcv = 1.1.5

gcc = 5.3.0


## Experiments

each abltion experiment is repeated 10 times under same condition

### 2021.4.23

#### split1

001-CCTV1, 019-fenghuangweishi, 011-history_channel

| | basline | +branch | +attention |
|:---:|:---:|:---:|:---:|
| min | 0.2641 | 0.2889(+0.0248) | 0.3171(+0.053) |
| max | 0.3847 | 0.4463(+0.0616) | 0.5280(+0.1433) |
| avg | 0.3080 | 0.3514(+0.0434) | 0.4422(+0.1342) |

### 2021.4.24

#### split2

undo

## undo experiments

- [ ] problem of score threshold in testing
- [ ] testing result is unstable
- [ ] difference between domain adaptation
- [ ] is our module really lead to the improvments?
- [ ] comparison with other few-shot detection method 
- [ ] data augmentation
