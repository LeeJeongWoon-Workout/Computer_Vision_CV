'''----------------------------------------------------------------------------------------Setting'''

!git clone https://github.com/ultralytics/yolov3
!cd yolov3;pip install -qr requirements.txt

import torch
from IPython.display import Image, clear_output  # to display images

clear_output()
print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")


'''----------------------------------------------------------------------------------------Dataset DownLoad'''

!mkdir /content/data
!wget https://public.roboflow.com/ds/259TpwzwWY?key=s73jV45MO8
  
!unzip -qq '/content/259TpwzwWY?key=s73jV45MO8' -d '/content/data'

!mkdir /content/pk;
!cd /content/pk; mkdir images; mkdir labels;
!cd /content/pk/images; mkdir train; mkdir val
!cd /content/pk/labels; mkdir train; mkdir val


'''-----------------------------------------------------------------------------------------xmlfile to anno_txt_file'''

import glob
import xml.etree.ElementTree as ET
import os, sys 
from google.colab import drive 
import shutil

# 1개의 voc xml 파일을 Yolo 포맷용 txt 파일로 변경하는 함수 
def xml_to_txt(input_xml_file, output_txt_file):
  # ElementTree로 입력 XML파일 파싱. 
  tree = ET.parse(input_xml_file)
  root = tree.getroot()
  img_node = root.find('size')
  # img_node를 찾지 못하면 종료
  if img_node is None:
    return None
  # 원본 이미지의 너비와 높이 추출. 
  img_width = int(img_node.find('width').text)
  img_height = int(img_node.find('height').text)

  # xml 파일내에 있는 모든 object Element를 찾음. 
  value_str = None
  with open(output_txt_file, 'w') as output_fpointer:
    for obj in root.findall('object'):
        # bndbox를 찾아서 좌상단(xmin, ymin), 우하단(xmax, ymax) 좌표 추출. 
        xmlbox = obj.find('bndbox')
        x1 = int(xmlbox.find('xmin').text)
        y1 = int(xmlbox.find('ymin').text)
        x2 = int(xmlbox.find('xmax').text)
        y2 = int(xmlbox.find('ymax').text)
        # 만약 좌표중에 하나라도 0보다 작은 값이 있으면 종료. 
        if (x1 < 0) or (x2 < 0) or (y1 < 0) or (y2 < 0):
          break
        class_id=obj.find('name').text

        if class_id == 'space-empty':
          class_id=0
        else:
          class_id=1
        # object_name과 원본 좌표를 입력하여 Yolo 포맷으로 변환하는 convert_yolo_coord()함수 호출. 
        cx_norm, cy_norm, w_norm, h_norm = convert_yolo_coord(img_width, img_height, x1, y1, x2, y2)
        # 변환된 yolo 좌표를 object 별로 출력 text 파일에 write
        value_str = ('{0} {1} {2} {3} {4}').format(class_id, cx_norm, cy_norm, w_norm, h_norm)
        output_fpointer.write(value_str+'\n')
        # debugging용으로 아래 출력
        #print(object_name, value_str)

# object_name과 원본 좌표를 입력하여 Yolo 포맷으로 변환
def convert_yolo_coord(img_width, img_height, x1, y1, x2, y2):
  # class_id는 CLASS_NAMES 리스트에서 index 번호로 추출. 
  # 중심 좌표와 너비, 높이 계산. 
  center_x = (x1 + x2)/2
  center_y = (y1 + y2)/2
  width = x2 - x1
  height = y2 - y1
  # 원본 이미지 기준으로 중심 좌표와 너비 높이를 0-1 사이 값으로 scaling
  center_x_norm = center_x / img_width
  center_y_norm = center_y / img_height
  width_norm = width / img_width
  height_norm = height / img_height

  return round(center_x_norm, 7), round(center_y_norm, 7), round(width_norm, 7), round(height_norm, 7)


'''---------------------------------------------------------------------------------------------------------data split'''

#pk 안에 images,labels 안에 각각 train,val을 넣어야 한다.

import pandas as pd
readlist_train=os.listdir('/content/data/train')
readlist_valid=os.listdir('/content/data/valid')

train_jpg_list=[x for x in readlist_train if 'xml' not in x]
train_xml_list=[x for x in readlist_train if 'xml' in x]

valid_jpg_list=[x for x in readlist_valid if 'xml' not in x]
valid_xml_list=[x for x in readlist_valid if 'xml' in x]

df1=pd.DataFrame({'train_jpg_file':train_jpg_list,'train_xml_file':train_xml_list})
df2=pd.DataFrame({'valid_jpg_file':valid_jpg_list,'valid_xml_file':valid_xml_list})

df1['img_name']=df1['train_jpg_file'].apply(lambda x:x[:x.rfind('.jpg')])
df1['img_filepath'] = '/content/data/train/' + df1['train_jpg_file']
df1['anno_filepath']='/content/data/train/'+df1['train_xml_file']

df2['img_name']=df2['valid_jpg_file'].apply(lambda x:x[:x.rfind('.jpg')])
df2['img_filepath']='/content/data/valid/' + df2['valid_jpg_file']
df2['anno_filepath']='/content/data/valid/'+df2['valid_xml_file']


import shutil

def make_yolo_anno_file(df, tgt_images_dir, tgt_labels_dir):
  for index, row in df.iterrows():
    src_image_path = row['img_filepath']
    src_label_path = row['anno_filepath']
    target_label_path = tgt_labels_dir + str(row['img_name'])+'.txt'
    shutil.copy(src_image_path, tgt_images_dir)
    xml_to_txt(src_label_path, target_label_path)

# train용 images와 labels annotation 생성. 
make_yolo_anno_file(df1, '/content/pk/images/train/', '/content/pk/labels/train/')
# val용 images와 labels annotation 생성. 
make_yolo_anno_file(df2, '/content/pk/images/val/', '/content/pk/labels/val/')


'''-----------------------------------------------------------------------------------------------train'''
drive.mount('/content/gdrive')

# soft link로 Google Drive Directory 연결. 
!ln -s /content/gdrive/My\ Drive/ /mydrive
!ls /mydrive

# Google Drive 밑에 Directory 생성. 이미 생성 되어 있을 시 오류 발생. 
!mkdir "/mydrive/ultra_workdir"


###  10번 미만 epoch는 좋은 성능이 안나옴. 최소 30번 이상 epoch 적용. 
!cd /content/yolov3; python train.py --img 640 --batch 16 --epochs 30 --data /content/pk/pk.yaml --weights yolov3.pt --project=/mydrive/ultra_workdir \
                                     --name pk --exist-ok 


'''------------------------------------------------------------------------------------------------eval'''

!cd yolov3; python test.py --weights /mydrive/ultra_workdir/pk/weights/best.pt  --data /content/pk/pk.yaml \
                           --project /content/data/output --name=test_result --exist-ok --img 640 --iou 0.65
