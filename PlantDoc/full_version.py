'''---------------------------------------------------------------------------------------------------setting'''

!pip install mmcv-full
!git clone https://github.com/open-mmlab/mmdetection.git
!cd mmdetection; python setup.py install


from mmdet.apis import init_detector, inference_detector
import mmcv


'''----------------------------------------------------------------------------------------------------Data Download'''
!mkdir /content/data
!wget https://public.roboflow.com/ds/S3NgfExtRv?key=HU2ThdxvCa
!unzip -qq '/content/S3NgfExtRv?key=HU2ThdxvCa' -d '/content/data'

'''----------------------------------------------------------------------------------------------------Config Setting'''
#custom_dataset
%cd /content
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset

@DATASETS.register_module(force=True)
class PlantDoc(CocoDataset):
  CLASSES = (
            "Apple Scab Leaf", "Apple leaf", "Apple rust leaf", "Bell_pepper leaf","Bell_pepper leaf spot",
"Blueberry leaf","Cherry leaf","Corn Gray leaf spot", "Corn leaf blight", "Corn rust leaf","Peach leaf",
"Potato leaf","Potato leaf early blight","Potato leaf late blight","Raspberry leaf","Soyabean leaf",
"Soybean leaf","Squash Powdery mildew leaf","Strawberry leaf","Tomato Early blight leaf","Tomato Septoria leaf spot",
 "Tomato leaf","Tomato leaf bacterial spot","Tomato leaf late blight", "Tomato leaf mosaic virus",
"Tomato leaf yellow virus","Tomato mold leaf", "Tomato two spotted spider mites leaf","grape leaf","grape leaf black rot")

  config_file = '/content/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = '/content/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

!cd /content/mmdetection; mkdir checkpoints
!wget -O /content/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
  
#config_model_download
from mmcv import Config

cfg = Config.fromfile(config_file)
print(cfg.pretty_text)
from mmdet.apis import set_random_seed

# dataset에 대한 환경 파라미터 수정. 
cfg.dataset_type = 'PlantDoc'
cfg.data_root = '/content/data/'

# train, val, test dataset에 대한 type, data_root, ann_file, img_prefix 환경 파라미터 수정. 
cfg.data.train.type = 'PlantDoc'
cfg.data.train.data_root = '/content/data/'
cfg.data.train.ann_file = 'train/_annotations.coco.json'
cfg.data.train.img_prefix = 'train'

cfg.data.val.type = 'PlantDoc'
cfg.data.val.data_root = '/content/data/'
cfg.data.val.ann_file = 'train/_annotations.coco.json'
cfg.data.val.img_prefix = 'train'

cfg.data.test.type = 'PlantDoc'
cfg.data.test.data_root = '/content/data/'
cfg.data.test.ann_file = 'test/_annotations.coco.json'
cfg.data.test.img_prefix = 'test'

# class의 갯수 수정. 
cfg.model.roi_head.bbox_head.num_classes = 30
# pretrained 모델
cfg.load_from = '/content/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# 학습 weight 파일로 로그를 저장하기 위한 디렉토리 설정. 
cfg.work_dir = './tutorial_exps'

# 학습율 변경 환경 파라미터 설정. 
cfg.optimizer.lr = 0.02 / 8
cfg.lr_config.warmup = None
cfg.log_config.interval = 10
cfg.runner.max_epochs = 10

# CocoDataset의 경우 metric을 bbox로 설정해야 함.(mAP아님. bbox로 설정하면 mAP를 iou threshold를 0.5 ~ 0.95까지 변경하면서 측정)
cfg.evaluation.metric = 'bbox'
cfg.evaluation.interval = 12
cfg.checkpoint_config.interval = 12

# 두번 config를 로드하면 lr_config의 policy가 사라지는 오류로 인하여 설정. 
cfg.lr_config.policy='step'
# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)


'''========================================================================================================train'''
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector

# train용 Dataset 생성. 
datasets = [build_dataset(cfg.data.train)]

#이 형식은 그대로 사용한다.
model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
model.CLASSES = datasets[0].CLASSES
print(model.CLASSES)
train_detector(model, datasets, cfg, distributed=False, validate=True)

'''------------------------------------------------------------------------------------------------------------eval'''
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
