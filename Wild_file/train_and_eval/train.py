from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector

# train용 Dataset 생성. 
#[] 안에 담아야 한다.
datasets = [build_dataset(cfg.data.train)]

#이 형식은 그대로 사용한다.
#모델을 생성할 때 인자들은 config 정보들이다.
#train_detector 의 인자로 모델,데이터셋,config,validate=True(val 데이터를 사용할 것인가.)
model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
model.CLASSES = datasets[0].CLASSES
print(model.CLASSES)
train_detector(model, datasets, cfg, distributed=False, validate=True)
