%cd /content
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset

@DATASETS.register_module(force=True)
class PKLotDataset(CocoDataset):
  CLASSES = ("movable-objects",'boat','car','dock','jetski','lift') 

