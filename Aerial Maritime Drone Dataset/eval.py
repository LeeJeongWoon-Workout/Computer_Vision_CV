'''result inference'''
from mmdet.apis import multi_gpu_test, single_gpu_test
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmdet.datasets import (build_dataloader, build_dataset)
import cv2
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

import torch
dataset = build_dataset(cfg.data.test)

data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1, #single_gpu_test를 사용하기 위해 gpu 값을 1로 초기화 해야 한다.
        workers_per_gpu=1,
        dist=False,
        shuffle=False)
#MMDetection Evaluation을 위한 함수    
model_ckpt = MMDataParallel(model, device_ids=[0])
#결과물
outputs = single_gpu_test(model_ckpt, data_loader, True, '/content/tutorial_exps', 0.3)

print('결과 outputs type:', type(outputs))
print('evalution 된 파일의 갯수:', len(outputs))
print('첫번째 evalutation 결과의 type:', type(outputs[0]))
print('첫번째 evaluation 결과의 CLASS 갯수:', len(outputs[0]))
print('첫번째 evaluation 결과의 CLASS ID 0의 type과 shape', type(outputs[0][0]), outputs[0][0].shape)
