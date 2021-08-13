from mmdet.apis import multi_gpu_test, single_gpu_test
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmdet.datasets import (build_dataloader, build_dataset)
import cv2
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

import torch
#evaluation을 하기 위한 과정
#1. dataset을 다시 만든다 []에 담지 않는다.
#2. data_loader를 생성한다.
#3. MMDataParallel로 모델을 평가용 모델을 만든다.
#4. output에 single_gpu_test(생성 모델,데이터 로더,,완료후 어디에 저장할 것인가)를 설정하도록 한다.
dataset = build_dataset(cfg.data.test)

data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1, #single_gpu_test를 사용하기 위해 gpu 값을 1로 초기화 해야 한다. 아니면 오류가 발생
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
