#!/usr/bin/env python
# coding: utf-8

import os
import time
import pynvml
import torch

pynvml.nvmlInit()
# 这里的0是GPU id
while True:
    time.sleep(30)
    localtime = time.asctime( time.localtime(time.time()) )
    print(localtime)
    for i in range(1,6):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print('gpu {} used {}'.format(i, meminfo.used / 1024 / 1024))
        if meminfo.used / 1024 / 1024 < 1000:
            print('get memory of gpu {}'.format(i))
            os.environ['CUDA_VISIBLE_DEVICES'] = str(i)
            a = torch.rand(2, 1024, 1024, 1024)
            a.cuda()




