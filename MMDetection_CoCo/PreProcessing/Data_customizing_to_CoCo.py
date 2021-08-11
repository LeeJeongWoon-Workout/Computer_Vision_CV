#Dataset Customizing Process
'''
PASCAL VOC형태의 BCCD Dataset를 Download 후 MS-COCO 형태로 변경한다.
-> 다운로드 받은 Dataset은 Pascal VOC 형태이므로 이를 별도의 유틸리티를 이용하여 MS-COCO 형태로 변환하도록 한다.
'''
#VOC를 COCO로 변환하는 package를 MMDetection 
!git clone https://github.com/Shenggan/BCCD_Dataset.git
!git clone https://github.com/yukkyo/voc2coco.git
#labels.txt 파일을 하나 생성한 후 적혈구,백혈구,혈소판 클래스 이름을 txt형식으로 저장한다.
#img_list=mmcv.list_from_file 를 사용하기 위함
import os
with open('/content/BCCD_Dataset/BCCD/labels.txt','w') as f:
  f.write('WBC\n')
  f.write('RBC\n')
  f.write('Platelets\n')

!cat /content/BCCD_Dataset/BCCD/labels.txt

# VOC를 COCO로 변환 수행. 학습/검증/테스트 용 json annotation을 생성. ->!git clone https://github.com/yukkyo/voc2coco.git 이용
 #xml 파일로 부터 정보를 받는다
 #거기서 어떤 걸 사용할 것인가
 #그리고 클래스 라벨을 부여해서
 #이렇게 반환할 것이다.
%cd voc2coco
!python voc2coco.py --ann_dir /content/BCCD_Dataset/BCCD/Annotations \
--ann_ids /content/BCCD_Dataset/BCCD/ImageSets/Main/train.txt \
--labels /content/BCCD_Dataset/BCCD/labels.txt \
--output /content/BCCD_Dataset/BCCD/train.json \
--ext xml

!python voc2coco.py --ann_dir /content/BCCD_Dataset/BCCD/Annotations \
--ann_ids /content/BCCD_Dataset/BCCD/ImageSets/Main/val.txt \
--labels /content/BCCD_Dataset/BCCD/labels.txt \
--output /content/BCCD_Dataset/BCCD/val.json \
--ext xml

!python voc2coco.py --ann_dir /content/BCCD_Dataset/BCCD/Annotations \
--ann_ids /content/BCCD_Dataset/BCCD/ImageSets/Main/test.txt \
--labels /content/BCCD_Dataset/BCCD/labels.txt \
--output /content/BCCD_Dataset/BCCD/test.json \
--ext xml


!cat /content/BCCD_Dataset/BCCD/train.json
# annotation json 파일을 잘 볼수 있는 jq 유틸리티 셋업. 
!sudo apt-get install jq
!jq . /content/BCCD_Dataset/BCCD/train.json > output.json
!tail -100 output.json
