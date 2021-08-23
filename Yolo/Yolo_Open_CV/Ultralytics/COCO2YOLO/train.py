!wget -O /content/bccd/bccd.yaml https://raw.githubusercontent.com/chulminkw/DLCV/master/data/util/bccd.yaml
  
# Google Drive 접근을 위한 Mount 적용. 
import os, sys 
from google.colab import drive 

drive.mount('/content/gdrive')


# soft link로 Google Drive Directory 연결. 
!ln -s /content/gdrive/My\ Drive/ /mydrive
!ls /mydrive
# Google Drive 밑에 Directory 생성. 이미 생성 되어 있을 시 오류 발생. 
!mkdir "/mydrive/ultra_workdir"


###  10번 미만 epoch는 좋은 성능이 안나옴. 최소 30번 이상 epoch 적용. large 모델 적용 시 batch size가 8보다 클 경우 colab에서 memory 부족 발생.
### 혈소판의 경우 상대적으로 mAP:0.5~0.95 Detection 성능이 좋지 못함. 백혈구 만큼 학습데이터가 적은것도 이유지만, Object 사이즈가 상대적으로 작음.   
!cd /content/yolov5; python train.py --img 640 --batch 8 --epochs 30 --data /content/bccd/bccd.yaml --weights yolov5l.pt \
                                     --project=/mydrive/ultra_workdir --name bccd --exist-ok 


from collections import Counter

anno_list = train_yolo_converter.labels['annotations']
category_list = [x['category_id'] for x in anno_list]

Counter(category_list)
