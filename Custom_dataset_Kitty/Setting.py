# mmcv를 위해서 mmcv-full을 먼저 설치해야 함. 
!pip install mmcv-full
# mmdetection 설치 
!git clone https://github.com/open-mmlab/mmdetection.git
!cd mmdetection; python setup.py install
#내 드라이브에 mmdetection directory를 생성한 후 가져온 깃헙 정보를 python setup 으로 install 한다.


# 아래를 수행하기 전에 kernel을 restart 해야 함. 
from mmdet.apis import init_detector, inference_detector
import mmcv

!wget https://download.openmmlab.com/mmdetection/data/kitti_tiny.zip
!unzip kitti_tiny.zip > /dev/null
