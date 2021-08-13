#custom_dataset
%cd /content
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset

@DATASETS.register_module(force=True)
class SmokeDataset(CocoDataset):
  CLASSES = ('Smoke','smoke') 
