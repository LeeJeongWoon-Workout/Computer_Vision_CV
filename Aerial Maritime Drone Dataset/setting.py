!pip install mmcv-full
#mmdetection.git의 내용들을 모두 가져와서
!git clone https://github.com/open-mmlab/mmdetection.git
#python의 형태로 다운로드를 한다.
!cd mmdetection; python setup.py install

!mkdir /content/data
!mkdir /content/data/train
!mkdir /content/data/test
!mkdir /content/data/valid

# kernel-restart
# 아래를 수행하기 전에 kernel을 restart 해야 함. 
from mmdet.apis import init_detector, inference_detector
import mmcv
