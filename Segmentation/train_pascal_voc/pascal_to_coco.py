!wget http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
!tar -xvf VOCtrainval_06-Nov-2007.tar > /dev/null 2>&1

!git clone https://github.com/ISSResearch/Dataset-Converters.git
!cd Dataset-Converters; pip install -r requirements.txt

!mkdir /content/coco_output
!cd Dataset-Converters;python convert.py --input-folder /content/VOCdevkit/VOC2007 --output-folder /content/coco_output \
                  --input-format VOCSEGM --output-format COCO --copy

!pip install opencv-python==4.1.2.30
