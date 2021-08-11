'''환경 설정'''
# 환경 Setting -> 고정 틀 그냥 씁시다.
!pip install mmcv-full
#mmdetection.git의 내용들을 모두 가져와서
!git clone https://github.com/open-mmlab/mmdetection.git
#python의 형태로 다운로드를 한다.
!cd mmdetection; python setup.py install

# kernel-restart
# 아래를 수행하기 전에 kernel을 restart 해야 함. 
from mmdet.apis import init_detector, inference_detector
import mmcv

'''Dataset customizing'''
#Dataset Customizing Process
'''
PASCAL VOC형태의 BCCD Dataset를 Download 후 MS-COCO 형태로 변경한다.
-> 다운로드 받은 Dataset은 Pascal VOC 형태이므로 이를 별도의 유틸리티를 이용하여 MS-COCO 형태로 변환하도록 한다.
'''
#VOC를 COCO로 변환하는 package를 MMDetection 
!git clone https://github.com/Shenggan/BCCD_Dataset.git
!git clone https://github.com/yukkyo/voc2coco.git
#labels.txt 파일을 하나 생성한 후 적혈구,백혈구,혈소판 클래스 이름을 txt형식으로 저장한다.
#img_list=mmcv.list_from_file 를 사용하기 위함
import os
with open('/content/BCCD_Dataset/BCCD/labels.txt','w') as f:
  f.write('WBC\n')
  f.write('RBC\n')
  f.write('Platelets\n')
'''-----------------------------------------------------------------------------------------------------------------------------'''
!cat /content/BCCD_Dataset/BCCD/labels.txt
# VOC를 COCO로 변환 수행. 학습/검증/테스트 용 json annotation을 생성. ->!git clone https://github.com/yukkyo/voc2coco.git 이용
 #xml 파일로 부터 정보를 받는다
 #거기서 어떤 걸 사용할 것인가
 #그리고 클래스 라벨을 부여해서
 #이렇게 반환할 것이다.
%cd voc2coco
!python voc2coco.py --ann_dir /content/BCCD_Dataset/BCCD/Annotations \
--ann_ids /content/BCCD_Dataset/BCCD/ImageSets/Main/train.txt \
--labels /content/BCCD_Dataset/BCCD/labels.txt \
--output /content/BCCD_Dataset/BCCD/train.json \
--ext xml

!python voc2coco.py --ann_dir /content/BCCD_Dataset/BCCD/Annotations \
--ann_ids /content/BCCD_Dataset/BCCD/ImageSets/Main/val.txt \
--labels /content/BCCD_Dataset/BCCD/labels.txt \
--output /content/BCCD_Dataset/BCCD/val.json \
--ext xml

!python voc2coco.py --ann_dir /content/BCCD_Dataset/BCCD/Annotations \
--ann_ids /content/BCCD_Dataset/BCCD/ImageSets/Main/test.txt \
--labels /content/BCCD_Dataset/BCCD/labels.txt \
--output /content/BCCD_Dataset/BCCD/test.json \
--ext xml

!cat /content/BCCD_Dataset/BCCD/train.json
# annotation json 파일을 잘 볼수 있는 jq 유틸리티 셋업. 
!sudo apt-get install jq
!jq . /content/BCCD_Dataset/BCCD/train.json > output.json
!tail -100 output.json

'''-----------------------------------------------------------------------------------------------------------------------------'''

#CocoDataset 클래스를 활용하여 BCCD Dataset을 로딩하기 -> CoCoDataset 형식이면 매우 쉬워진다.
%cd /content
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset

@DATASETS.register_module(force=True)
class BCCDDataset(CocoDataset):
  CLASSES = ('WBC', 'RBC', 'Platelets') 


'''model_download and config setting'''

#Model Download Config Setting
config_file = '/content/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = '/content/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

!cd /content/mmdetection; mkdir checkpoints
!wget -O /content/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth

 
from mmcv import Config

cfg = Config.fromfile(config_file)
print(cfg.pretty_text)
from mmdet.apis import set_random_seed

# dataset에 대한 환경 파라미터 수정. 
cfg.dataset_type = 'BCCDDataset'
cfg.data_root = '/content/BCCD_Dataset/BCCD/'

# train, val, test dataset에 대한 type, data_root, ann_file, img_prefix 환경 파라미터 수정. 
cfg.data.train.type = 'BCCDDataset'
cfg.data.train.data_root = '/content/BCCD_Dataset/BCCD/'
cfg.data.train.ann_file = 'train.json'
cfg.data.train.img_prefix = 'JPEGImages'

cfg.data.val.type = 'BCCDDataset'
cfg.data.val.data_root = '/content/BCCD_Dataset/BCCD/'
cfg.data.val.ann_file = 'val.json'
cfg.data.val.img_prefix = 'JPEGImages'

cfg.data.test.type = 'BCCDDataset'
cfg.data.test.data_root = '/content/BCCD_Dataset/BCCD/'
cfg.data.test.ann_file = 'test.json'
cfg.data.test.img_prefix = 'JPEGImages'

# class의 갯수 수정. 
cfg.model.roi_head.bbox_head.num_classes = 3
# pretrained 모델
cfg.load_from = '/content/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# 학습 weight 파일로 로그를 저장하기 위한 디렉토리 설정. 
cfg.work_dir = './tutorial_exps'

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

'''-----------------------------------------------------------------------------------------------------------------------------'''


'''train'''
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector

# train용 Dataset 생성. 
datasets = [build_dataset(cfg.data.train)]

model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
model.CLASSES = datasets[0].CLASSES
print(model.CLASSES)
train_detector(model, datasets, cfg, distributed=False, validate=True)


'''result inference'''
from mmdet.apis import multi_gpu_test, single_gpu_test
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmdet.datasets import (build_dataloader, build_dataset)
import cv2
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

img = cv2.imread('/content/BCCD_Dataset/BCCD/JPEGImages/BloodImage_00007.jpg')

model.cfg = cfg

result = inference_detector(model, img)
show_result_pyplot(model, img, result)


'''Evaluation'''
import torch
dataset = build_dataset(cfg.data.test)
#파이토치 기반 데이터 로더 생성
#MMdetection을 eval하기 위해서는 MDataParallel로 만든 평가용 포델과,파이토치 데이터 로더를
#single_gpu_test에 집어 넣어야 한다.
'''
설계
1.data loader를 생성한다. (gpu값을 모두 1로 초기화 해야 single_gpu_test가 작동한다.)
2.MMDataParallel를 통해 eval용 model를 하나 더 만든다.
3.single_gpu_test(model,data_loader,True,~저장위치)를 집어 넎느다.
'''
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
'''-----------------------------------------------------------------------------------------------------------------------------'''
'''
1. 파일 이름을 os.listdir 로 한번에 저장한다.
2. file_path를 만든다.
3. img_array를 생성한다.
'''

path_dir='/content/tutorial_exps'
file_list = os.listdir(path_dir)
file_path=['/content/tutorial_exps/'+x for x in file_list]

img_array=[cv2.imread(x) for x in file_path]


#이미지 출력 함수 subplots 사용
def show_detected_images(img_arrays, ncols=50):
    figure, axs = plt.subplots(figsize=(22, 6), nrows=1, ncols=ncols)
    for i in range(ncols):
      axs[i].imshow(img_arrays[i])

for i in range(50):
  show_detected_images(img_array[i:5*(i+1)], ncols=5)
