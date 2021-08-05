import torch
print(torch.__version__)

!pip install mmcv-full

#우리가 사용할 MMDetection의 모델은 PyTorch 기반으로 짜여져 있기 때문에 Pytorch를 다운받을 겁니다.
#또한 MMDetection을 사용하기 위해서는 mmcv를 다운받아야 하는데 이 mmcv가 용량이 커 약 10분의 시간이 소요됩니다.
