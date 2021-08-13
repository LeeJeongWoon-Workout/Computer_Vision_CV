!pip install mmcv-full
!git clone https://github.com/open-mmlab/mmdetection.git
!cd mmdetection; python setup.py install


from mmdet.apis import init_detector, inference_detector
import mmcv

!mkdir /content/data
!mkdir /content/data/annotations
!mkdir /content/data/train
!mkdir /content/data/test
!mkdir /content/data/valid

!cd /content/data/train && unzip train.zip 
!cd /content/data/test && unzip test.zip 
!cd /content/data/valid && unzip valid.zip 

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset

@DATASETS.register_module(force=True)
class ChessDataset(CocoDataset):
  CLASSES = ("bishop","black-bishop","black-king","black-knight","black-pawn","black-queen","black-rook","white-bishop","white-king","white-knight","white-pawn","white-queen","white-rook") 
  
  
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
cfg.dataset_type = 'ChessDataset'
cfg.data_root = '/content/data/'

# train, val, test dataset에 대한 type, data_root, ann_file, img_prefix 환경 파라미터 수정. 
cfg.data.train.type = 'ChessDataset'
cfg.data.train.data_root = '/content/data/'
cfg.data.train.ann_file = 'annotations/train_annotations.coco.json'
cfg.data.train.img_prefix = 'train'

cfg.data.val.type = 'ChessDataset'
cfg.data.val.data_root = '/content/data/'
cfg.data.val.ann_file = 'annotations/valid_annotations.coco.json'
cfg.data.val.img_prefix = 'valid'

cfg.data.test.type = 'ChessDataset'
cfg.data.test.data_root = '/content/data/'
cfg.data.test.ann_file = 'annotations/test_annotations.coco.json'
cfg.data.test.img_prefix = 'test'

# class의 갯수 수정. 
cfg.model.roi_head.bbox_head.num_classes = 13
# pretrained 모델
cfg.load_from = '/content/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# 학습 weight 파일로 로그를 저장하기 위한 디렉토리 설정. 
cfg.work_dir = './tutorial_exps'
cfg.runner.max_epochs = 5

# 학습율 변경 환경 파라미터 설정. 
cfg.optimizer.lr = 0.02 / 8
cfg.lr_config.warmup = None
cfg.log_config.interval = 10

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



import numpy as np
import cv2
def get_detected_img(model, img_array,  score_threshold=0.3, is_print=True):
  # 인자로 들어온 image_array를 복사. 
  draw_img = img_array.copy()
  bbox_color=[
              (100,0,0),(0,100,0),(0,0,100),(100,100,0),(100,0,100),(0,100,100),(100,50,50),(50,100,50),(50,50,100),(30,50,70),(50,30,70),(50,70,30),(70,50,30)
  ]
  text_color=(0, 0, 255)


  results = inference_detector(model, img_array)
  labels_to_names_seq= ("bishop","black-bishop","black-king","black-knight","black-pawn","black-queen","black-rook","white-bishop","white-king","white-knight","white-pawn","white-queen","white-rook")

  for result_ind, result in enumerate(results):
    # 개별 2차원 array의 row size가 0 이면 해당 Class id로 값이 없으므로 다음 loop로 진행. 
    if len(result) == 0 or result_ind==0: 
      #산불이 아닌 것을 검출할 필요가 없으므로 0번째 인덱스는 거른다.
      continue
    
    # 2차원 array에서 5번째 컬럼에 해당하는 값이 score threshold이며 이 값이 함수 인자로 들어온 score_threshold 보다 낮은 경우는 제외. 
    result_filtered = result[np.where(result[:, 4] > score_threshold)]
    
    # 해당 클래스 별로 Detect된 여러개의 오브젝트 정보가 2차원 array에 담겨 있으며, 이 2차원 array를 row수만큼 iteration해서 개별 오브젝트의 좌표값 추출. 
    for i in range(len(result_filtered)):
      # 좌상단, 우하단 좌표 추출. 
      left = int(result_filtered[i, 0])
      top = int(result_filtered[i, 1])
      right = int(result_filtered[i, 2])
      bottom = int(result_filtered[i, 3])
      caption = "{}: {:.4f}".format(labels_to_names_seq[result_ind], result_filtered[i, 4])
      cv2.rectangle(draw_img, (left, top), (right, bottom), color=bbox_color[result_ind], thickness=2)
      cv2.putText(draw_img, caption, (int(left), int(top - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
      if is_print:
        print(caption)

  return draw_img



image=cv2.imread('')
image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
model.cfg=cfg
result=get_detected_img(model,imagge)
plt.imshow(result)
plt.show()
