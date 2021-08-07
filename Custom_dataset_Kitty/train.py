#우리는 이전에는 그저 pretrained 된 모델을  inference_detector,init_detector를 이용해 그대로 사용했다.
#하지만 이제부터 우리는 원하는 모델들과 클래스 종류를 가지고 우리가 원하는 방향으로 모델을 바꿀 수 있다. 즉 범용성이 증가한 것이다.
#그것을 위해 build_dataset 과 build_detector train_detector 함수의 사용을 유심히 관찰해 보자.

from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
#내가 가진 자료로 모델을 pre_trained 하는 과정
# train용 Dataset 생성. 
#config의 train 데이터의 정보를 기반으로 훈련 데이터를 생성한다.
datasets = [build_dataset(cfg.data.train)]

#이제 pretrained 모델을 build_detector로 만든다 cfg 에 사용할 모델(res) 과 훈련 config 테스트 config 정보를 입력하자. 훈련과 테스트를 통해 모델이 자동으로 weight opti를 할 것
model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
model.CLASSES = datasets[0].CLASSES
#class는 4개
# 주의, config에 pretrained 모델 지정이 상대 경로로 설정됨 cfg.load_from = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
# 아래와 같이 %cd mmdetection 지정 필요. 
 
%cd mmdetection 

mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
# epochs는 config의 runner 파라미터로 지정됨. 기본 12회 
train_detector(model, datasets, cfg, distributed=False, validate=True)

from mmdet.apis import inference_detector, init_detector, show_result_pyplot

# BGR Image 사용 
img = cv2.imread('/content/kitti_tiny/training/image_2/000068.jpeg')

model.cfg = cfg
#kitty-tiny 데이터를 통해 훈련한 모델을 가지고 image object_detection을 수행해 보자.
result = inference_detector(model, img)
show_result_pyplot(model, img, result)
#mmdet show_result_pyplot 으로 결과를 보자.
