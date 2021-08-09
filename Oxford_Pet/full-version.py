# csv 파일로 test,val 메타 파일 분리 -> xml extractor ->  custormdataset -> 모델 다운, config 기입 -> tarin -> 적용

'''-----------------------------------------------------------------------------------------------------------------------------------------------: library setting'''

!pip install mmcv-full
!git clone https://github.com/open-mmlab/mmdetection.git
!cd mmdetection; python setup.py install
'''-------------------------------------------------------------------------------------------------------------------------------------------------: kernel restart'''

# 런타임->런타임 다시 시작 후 아래 수행. 
from mmdet.apis import init_detector, inference_detector
import mmcv
'''----------------------------------------------------------------------------------------------------------------------------------------------: unzip oxford dataset'''

!wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
!wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
  
# /content/data 디렉토리를 만들고 해당 디렉토리에 다운로드 받은 압축 파일 풀기.
!mkdir /content/data
!tar -xvf images.tar.gz -C /content/data
!tar -xvf annotations.tar.gz -C /content/data



!wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
!wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz

# /content/data 디렉토리를 만들고 해당 디렉토리에 다운로드 받은 압축 파일 풀기.
!mkdir /content/data  #pet 데이터를 보관한 directory 생성
!tar -xvf images.tar.gz -C /content/data  #tar -xvf(압축을 푼다) ~(~파일을) -C/content/data (-C ~ 위치에)
!tar -xvf annotations.tar.gz -C /content/data


'''------------------------------------------------------------------------------------------------------------------------------------------------------: xml extractor'''

#xml 파일에서 좌표와 파일 이름을 추춣하는 함수
'''
설계
1.get_bboxes_from_xml(anno_dir,xml_file)
osp.join 으로 anno_dir,xml_file을 합쳐 절대 경로를 만든다.
2. 절대 경로를 기준으로 tree,root 객체를 생성한다.
3. bbox_names,bboxes 리스트 객체 생성
4. loop (root.findall('object))
5. 클레스 명은 파일명에 있으므로 xml_file[:xml_file.rfind('_')] 문자열 함수 이용
6. 나머지 'bndbox'에서 좌표 추출
7. bbox_names,bboxes 반환
'''

import xml.etree.ElementTree as ET
import os.path as osp

def get_bboxes_from_xml(anno_dir,xml_file):
  anno_xml_file=osp.join(anno_dir,xml_file) #절대 경로
  tree=ET.parse(anno_xml_file)
  root=tree.getroot()
  bbox_names=[]
  bboxes=[]

  for obj in root.findall('object'):
    #obj.find('name).text는 cat이나 dog를 반환
    #object 클래스명은 파일명에서 추출
    bbox_name=xml_file[:xml_file.rfind('_')]

    xmlbox=obj.find('bndbox')

    x1=int(xmlbox.find('xmin').text)
    y1=int(xmlbox.find('ymin').text)
    x2=int(xmlbox.find('xmax').text)
    y2=int(xmlbox.find('ymax').text)

    bbox_names.append(bbox_name)
    bboxes.append([x1,y1,x2,y2])
  
  return bbox_names,bboxes
  '''--------------------------------------------------------------: train,val ann_file generate'''

  '''현재 trainval 에 다 모여 있기 때문에 pandas를 이용해 
csv 파일로 만든후 sklear.model_selection으로 나눈 다음 to_csv를 통해
다시 train 용 val 용 데이터를 담는 txt 메타 파일을 만들 것이다.'''
'''
과정
1.pd.read_csv(파일 경로,무슨 기준으로 정보들을 나눌 것인가,헤더,열 이름들 리스트로 입력)
2.apply(lambda로) class_name을 담는 새로운 column 생성
3. sklearn.model_selection, train_test_split로 train_df와 val_df 를 만든다.
4. sort_values(by='img_name')
5. ~['img_name'].to_csv('./data/train.txt',~) 로 훈련용,val용 txt 파일을 만든다.
'''

import pandas as pd

pet_df=pd.read_csv('./data/annotations/trainval.txt',sep=' ',header=None,names=['img_name','class_id','etc1','ect2'])
pet_df['class_name']=pet_df['img_name'].apply(lambda x:x[:x.rfind('_')])
pet_df.head()
pet_df=pet_df.sort_values(by='img_name')

from sklearn.model_selection import train_test_split

train_df,val_df=train_test_split(pet_df,test_size=0.1)
#train옹 valid용 메타파일을 따로 만드는 과정
train_df['img_name'].to_csv('./data/train.txt',sep=' ',header=False,index=False)
val_df['img_name'].to_csv('./data/val.txt',sep=' ',header=False,index=False)

!echo 'train list #####';cat ./data/train.txt
!echo 'val list #####'; cat ./data/val.txt

'''----------------------------------------------------------------------------------------------------------------------------------------------: custom dataset'''

import copy
import os.path as osp

import mmcv
import numpy as np
import cv2

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset

import xml.etree.ElementTree as ET

PET_CLASSES = pet_df['class_name'].unique().tolist()

  #ann 어떤걸 train,validation으로 쓸 것인지 알려주는 것 오직 하나만 있어야 한다.
  #prefix: 이미지가 들어 있는 파일 -> custom dataset에서 def__init__ 을 거쳐 data_root와 join 되어 절대 경로가 된다.
  #data_root= 거기까지 가기위한 루트들

  '''mid dataset load_annotation 설계 과정
  
  1. 클레스 별 번호 생성
  2. ann 파일로 부터 받은 이미지 파일을 루프에 넣어 파일이름,cv2를 이용한 너비,높이
  3. 클래스 명과 좌표를 얻기 위해서는 xml 파일을 훑어야 하므로 label
  prefix를 설정한다. label_prefix는 images를 annotations로 바꾼다.
  4. xml파일에 접근하기 위한 절대 경로를 설정한다.
  5. 오류가 발생하지 않기 위해 
        if not osp.exists(anno_xml_file):
        continue
  6.get_bboxes_from_(prefix,나머지 파일 이름) 을 실행한다.
  7. 나머지 이하동일 (넘파이로 바꾸는 것 잊지 않기)
  '''


@DATASETS.register_module(force=True)
class PetDataset(CustomDataset):
  CLASSES = PET_CLASSES

  # annotation에 대한 모든 파일명을 가지고 있는 텍스트 파일을 __init__(self, ann_file)로 입력 받고, 
  # 이 self.ann_file이 load_annotations()의 인자로 입력
  def load_annotations(self, ann_file):
    cat2label = {k:i for i, k in enumerate(self.CLASSES)}
    image_list = mmcv.list_from_file(self.ann_file)
    # 포맷 중립 데이터를 담을 list 객체
    data_infos = []

    for image_id in image_list:
      # self.img_prefix는 images 가 입력될 것임. 
      filename = '{0:}/{1:}.jpg'.format(self.img_prefix, image_id)
      # 원본 이미지의 너비, 높이를 image를 직접 로드하여 구함. 
      image = cv2.imread(filename)
      height, width = image.shape[:2]
      # 개별 image의 annotation 정보 저장용 Dict 생성. key값 filename에는 image의 파일명만 들어감(디렉토리는 제외)
      data_info = {'filename': filename,
                  'width': width, 'height': height}
      # 개별 annotation XML 파일이 있는 서브 디렉토리의 prefix 변환. 
      label_prefix = self.img_prefix.replace('images', 'annotations')
      
      # 개별 annotation XML 파일을 1개 line 씩 읽어서 list 로드. annotation XML파일이 xmls 밑에 있음에 유의
      anno_xml_file = osp.join(label_prefix, 'xmls/'+str(image_id)+'.xml')
      # 메타 파일에는 이름이 있으나 실제로는 존재하지 않는 XML이 있으므로 이는 제외. 
      if not osp.exists(anno_xml_file):
          continue
      
      # get_bboxes_from_xml() 를 이용하여 개별 XML 파일에 있는 이미지의 모든 bbox 정보를 list 객체로 생성. 
      anno_dir = osp.join(label_prefix, 'xmls')
      bbox_names, bboxes = get_bboxes_from_xml(anno_dir, str(image_id)+'.xml')
      #print('#########:', bbox_names)
                  
      gt_bboxes = []
      gt_labels = []
      gt_bboxes_ignore = []
      gt_labels_ignore = []
        
      # bbox별 Object들의 class name을 class id로 매핑. class id는 tuple(list)형의 CLASSES의 index값에 따라 설정
      for bbox_name, bbox in zip(bbox_names, bboxes):
        # 만약 bbox_name이 클래스명에 해당 되면, gt_bboxes와 gt_labels에 추가, 그렇지 않으면 gt_bboxes_ignore, gt_labels_ignore에 추가
        # bbox_name이 CLASSES중에 반드시 하나 있어야 함. 안 그러면 FILTERING 되므로 주의 할것. 
        if bbox_name in cat2label:
            gt_bboxes.append(bbox)
            # gt_labels에는 class id를 입력
            gt_labels.append(cat2label[bbox_name])
        else:
            gt_bboxes_ignore.append(bbox)
            gt_labels_ignore.append(-1)
      
      # 개별 image별 annotation 정보를 가지는 Dict 생성. 해당 Dict의 value값을 np.array형태로 bbox의 좌표와 label값으로 생성. 
      data_anno = {
        'bboxes': np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
        'labels': np.array(gt_labels, dtype=np.long),
        'bboxes_ignore': np.array(gt_bboxes_ignore, dtype=np.float32).reshape(-1, 4),
        'labels_ignore': np.array(gt_labels_ignore, dtype=np.long)
      }
      
      # image에 대한 메타 정보를 가지는 data_info Dict에 'ann' key값으로 data_anno를 value로 저장. 
      data_info.update(ann=data_anno)
      # 전체 annotation 파일들에 대한 정보를 가지는 data_infos에 data_info Dict를 추가
      data_infos.append(data_info)
      #print(data_info)

    return data_infos
'''----------------------------------------------------------------------------------------------------------------------------------: model download and config revise '''

      
config_file = './mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = './mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

!cd mmdetection; mkdir checkpoints
!wget -O ./mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth

from mmcv import Config

cfg = Config.fromfile(config_file)
print(cfg.pretty_text)

from mmdet.apis import set_random_seed

# dataset에 대한 환경 파라미터 수정. 
cfg.dataset_type = 'PetDataset'
cfg.data_root = '/content/data/'

# train, val, test dataset에 대한 type, data_root, ann_file, img_prefix 환경 파라미터 수정. 
cfg.data.train.type = 'PetDataset'
cfg.data.train.data_root = '/content/data/'
cfg.data.train.ann_file = 'train.txt'
cfg.data.train.img_prefix = 'images'

cfg.data.val.type = 'PetDataset'
cfg.data.val.data_root = '/content/data/'
cfg.data.val.ann_file = 'val.txt'
cfg.data.val.img_prefix = 'images'

# class의 갯수 수정. 
cfg.model.roi_head.bbox_head.num_classes = 37
# pretrained 모델
cfg.load_from = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# 학습 weight 파일로 로그를 저장하기 위한 디렉토리로 구글 Drive 설정. 
cfg.work_dir = '/mydrive/pet_work_dir'

# 학습율 변경 환경 파라미터 설정. 
cfg.optimizer.lr = 0.02 / 8
cfg.lr_config.warmup = None
cfg.log_config.interval = 5

cfg.runner.max_epochs = 5

# 평가 metric 설정. 
cfg.evaluation.metric = 'mAP'
# 평가 metric 수행할 epoch interval 설정. 
cfg.evaluation.interval = 5
# 학습 iteration시마다 모델을 저장할 epoch interval 설정. 
cfg.checkpoint_config.interval = 5

# 학습 시 Batch size 설정(단일 GPU 별 Batch size로 설정됨)
cfg.data.samples_per_gpu = 4

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)
# 두번 config를 로드하면 lr_config의 policy가 사라지는 오류로 인하여 설정. 
cfg.lr_config.policy='step'
# We can initialize the logger for training and have a look
# at the final config used for training
print(f'Config:\n{cfg.pretty_text}')
'''-----------------------------------------------------------------------------------------------------------------------: train'''

from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector

# train용 Dataset 생성. 
datasets = [build_dataset(cfg.data.train)]

%cd mmdetection

model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
model.CLASSES = datasets[0].CLASSES

mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
# epochs는 config의 runner 파라미터로 지정됨. 기본 12회 
train_detector(model, datasets, cfg, distributed=False, validate=True)
'''------------------------------------------------------------------------------------------------------------------: visualization'''

'''design
1. img_path열을 새롭게 만든다.
2. val_paths=val_df[val_df['img_path'].str.contains('~')]['img_path'].values  ~을 포함하는 이미지 주소를 numpy로 보관한다.
3. val_imgs=[cv2.imread(x) for x in val_paths] 로 주소의 이미지들을 cv2 로 numpy 화 한다.
4. get_detected_image 함수 생성
5. show_detected_image 함수 생성
'''
#이미지 주소 colum을 따로 만들어서 저장한다.
val_df['img_path'] = '/content/data/images/' + val_df['image_id'] + '.jpg'
val_df.head()

#val_df csv 파일에서 'img_path' 부분중 'Abyssinian'을 포함하는 
val_paths = val_df[val_df['img_path'].str.contains('Abyssinian')]['img_path'].values
val_imgs = [cv2.imread(x) for x in val_paths]

PET_CLASSES = pet_df['class_name'].unique().tolist()
labels_to_names_seq = {i:k for i, k in enumerate(PET_CLASSES)}


'''get_detected_img(model,img_array,socre_threshold,is_print) 함수 설계도
1. 이미지 복사,색깔설정, 모델과 이미지로 inference 실시 -> results는 37rows를 가진 numpy 37은 우리가 roi-head를 petdataset으로 조정하고 37개의 클래스로 설정하였음
2. enumerate(results) 루프 실행( 수행 첫 숫자는 이미지 클래스를 나타내고 result는 그 분류에 들어가는 좌표numpy)
3. result_filtered = result[np.where(result[:, 4] > score_threshold)]
4. result_filtered 루프를 실행하고 cv2.rentangle,caption,cv2.puttext 
'''


# model과 원본 이미지 array, filtering할 기준 class confidence score를 인자로 가지는 inference 시각화용 함수 생성. 
def get_detected_img(model, img_array,  score_threshold=0.3, is_print=True):
  # 인자로 들어온 image_array를 복사. 
  draw_img = img_array.copy()
  bbox_color=(0, 255, 0)
  text_color=(0, 0, 255)

  # model과 image array를 입력 인자로 inference detection 수행하고 결과를 results로 받음. 
  # results는 80개의 2차원 array(shape=(오브젝트갯수, 5))를 가지는 list. 
  results = inference_detector(model, img_array)

  # 80개의 array원소를 가지는 results 리스트를 loop를 돌면서 개별 2차원 array들을 추출하고 이를 기반으로 이미지 시각화 
  # results 리스트의 위치 index가 바로 COCO 매핑된 Class id. 여기서는 result_ind가 class id
  # 개별 2차원 array에 오브젝트별 좌표와 class confidence score 값을 가짐. 
  for result_ind, result in enumerate(results):
    # 개별 2차원 array의 row size가 0 이면 해당 Class id로 값이 없으므로 다음 loop로 진행. 
    if len(result) == 0:
      continue
    
    # 2차원 array에서 5번째 컬럼에 해당하는 값이 score threshold이며 이 값이 함수 인자로 들어온 score_threshold 보다 낮은 경우는 제외. 
    result_filtered = result[np.where(result[:, 4] > score_threshold)]
    
    # 해당 클래스 별로 Detect된 여러개의 오브젝트 정보가 2차원 array에 담겨 있으며, 이 2차원 array를 row수만큼 iteration해서 개별 오브젝트의 좌표값 추출. 
    for i in range(len(result_filtered)):
      # 좌상단, 우하단 좌표 추출. 
      left = int(result_filtered[i, 0])
      top = int(result_filtered[i, 1])
      right = int(result_filtered[i, 2])
      bottom = int(result_filtered[i, 3])
      caption = "{}: {:.4f}".format(labels_to_names_seq[result_ind], result_filtered[i, 4])
      cv2.rectangle(draw_img, (left, top), (right, bottom), color=bbox_color, thickness=2)
      cv2.putText(draw_img, caption, (int(left), int(top - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
      if is_print:
        print(caption)

  return draw_img

import matplotlib.pyplot as plt

img_arr = cv2.imread('/content/data/images/Abyssinian_88.jpg')
detected_img = get_detected_img(model, img_arr,  score_threshold=0.3, is_print=True)
# detect 입력된 이미지는 bgr임. 이를 최종 출력시 rgb로 변환 
detected_img = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 12))
plt.imshow(detected_img)


import matplotlib.pyplot as plt
import cv2
%matplotlib inline 

'''show_detected_images : 그래프화 함수
1. 모양,축 설정 figure,axis=plt.subplot(figsize=(),nrows=1,ncols=ncols)
2. 열루프 실행
3. 각 img_arrays별로 detected_img 디텍트 실행
4. cv2.cvtColor 로 BGR 를 RGB로 변경
5. axs[i].imshow(detected_img) -> 그래프 표현
'''

def show_detected_images(model, img_arrays, ncols=50):
    figure, axs = plt.subplots(figsize=(22, 6), nrows=1, ncols=ncols)
    for i in range(ncols):
      detected_img = get_detected_img(model, img_arrays[i],  score_threshold=0.5, is_print=True)
      detected_img = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)
      #detected_img = cv2.resize(detected_img, (328, 328))
      axs[i].imshow(detected_img)

        
show_detected_images(model_ckpt, val_imgs[:10], ncols=10)


val_paths = val_df['img_path'].values
val_imgs = [cv2.imread(x) for x in val_paths]

show_detected_images(model_ckpt, val_imgs[:5], ncols=5)
show_detected_images(model_ckpt, val_imgs[5:10], ncols=5)
show_detected_images(model_ckpt, val_imgs[10:15], ncols=5)
show_detected_images(model_ckpt, val_imgs[15:20], ncols=5)
show_detected_images(model_ckpt, val_imgs[20:25], ncols=5)
show_detected_images(model_ckpt, val_imgs[25:30], ncols=5)
show_detected_images(model_ckpt, val_imgs[30:35], ncols=5)
show_detected_images(model_ckpt, val_imgs[35:40], ncols=5)
show_detected_images(model_ckpt, val_imgs[40:45], ncols=5)
show_detected_images(model_ckpt, val_imgs[45:50], ncols=5)

