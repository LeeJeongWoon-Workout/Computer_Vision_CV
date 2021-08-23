import os

!git clone https://github.com/Shenggan/BCCD_Dataset.git
!git clone https://github.com/yukkyo/voc2coco.git

# colab 버전은 아래 명령어로 BCCD의 labels.txt를 생성합니다.  
with open('/content/BCCD_Dataset/BCCD/labels.txt', "w") as f:
    f.write("WBC\n")
    f.write("RBC\n")
    f.write("Platelets\n")

!cat /content/BCCD_Dataset/BCCD/labels.txt

# VOC를 COCO로 변환 수행. 학습/검증/테스트 용 json annotation을 생성. 
'''
--ann_dir : xml 파일 위치
--ann_ids : 사용할 이미지 이름
--labels : class 목록
--output : 어떻게 분출할 것인가.

'''
'''Pascal Voc vs CoCo.json format
1.Pascal Voc : 한 이미지에 한 labels(xml)
2.coco json train,test,valid 별로 이미지가 따로 분리되어 있고 그 전체 이미지에 대한 annotation이 json파일에 한번에 담겨 있음

'''


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

# annotation json 파일을 잘 볼수 있는 jq 유틸리티 셋업. 
!sudo apt-get install jq
!jq . /content/BCCD_Dataset/BCCD/train.json > output.json
!tail -100 output.json
