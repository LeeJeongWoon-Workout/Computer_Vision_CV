import copy
import os.path as osp

import mmcv
import numpy as np
import cv2

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset

import xml.etree.ElementTree as ET




@DATASETS.register_module(force=True)
class BloodDataset(CustomDataset):
  CLASSES = ('RBC','WBC','Platelets')
#3개의 클래스
  def load_annotations(self, ann_file):
    cat2label = {k:i for i, k in enumerate(self.CLASSES)}
    image_list = mmcv.list_from_file(self.ann_file)
    data_infos = []

    for image_id in image_list:
      filename = '{0:}/{1:}.jpg'.format(self.img_prefix, image_id)
      image = cv2.imread(filename)
      height, width = image.shape[:2]
      data_info = {'filename': filename,
                  'width': width, 'height': height}
      label_prefix = self.img_prefix.replace('JPEGImages', 'Annotations')
      
      anno_xml_file = osp.join(label_prefix,str(image_id)+'.xml')
      if not osp.exists(anno_xml_file):
          continue
      
      anno_dir = label_prefix
      bbox_names, bboxes = get_bboxes_from_xml(anno_dir, str(image_id)+'.xml')
                  
      gt_bboxes = []
      gt_labels = []
      gt_bboxes_ignore = []
      gt_labels_ignore = []
        
      for bbox_name, bbox in zip(bbox_names, bboxes):
        if bbox_name in cat2label:
            gt_bboxes.append(bbox)
            gt_labels.append(cat2label[bbox_name])
        else:
            gt_bboxes_ignore.append(bbox)
            gt_labels_ignore.append(-1)
      
      data_anno = {
        'bboxes': np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
        'labels': np.array(gt_labels, dtype=np.long),
        'bboxes_ignore': np.array(gt_bboxes_ignore, dtype=np.float32).reshape(-1, 4),
        'labels_ignore': np.array(gt_labels_ignore, dtype=np.long)
      }
      
  
      data_info.update(ann=data_anno)
 
      data_infos.append(data_info)
 

    return data_infos
