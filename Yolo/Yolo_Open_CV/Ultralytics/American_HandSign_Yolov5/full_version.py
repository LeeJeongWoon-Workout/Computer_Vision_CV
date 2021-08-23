'''-------------------------------------------------------------------------------model setting'''
!git clone https://github.com/ultralytics/yolov5
!cd yolov5;pip install -qr requirements.txt

'''-------------------------------------------------------------------------------data download'''
!mkdir /content/data
!wget https://public.roboflow.com/ds/Ns4lVVgZJ6?key=jxKTsyv9Ml
  
'''-------------------------------------------------------------------------------train'''
from google.colab import drive 


drive.mount('/content/gdrive')

# soft link로 Google Drive Directory 연결. 
!ln -s /content/gdrive/My\ Drive/ /mydrive
!ls /mydrive

# Google Drive 밑에 Directory 생성. 이미 생성 되어 있을 시 오류 발생. 
!mkdir "/mydrive/ultra_workdir"


###  10번 미만 epoch는 좋은 성능이 안나옴. 최소 30번 이상 epoch 적용. 
!cd /content/yolov5; python train.py --img 640 --batch 4 --epochs 30 --data /content/data.yaml --weights yolov5l.pt \
                                     --project=/mydrive/ultra_workdir --name screen --exist-ok 


'''-----------------------------------------------------------------------------------eval'''
!cd /content/yolov5;python detect.py --source /content/hand-sign.mp4 \
                            --weights /mydrive/ultra_workdir4/screen/weights/best.pt --conf 0.3 \
                            --project=/content/data/output --name=run_image --exist-ok --line-thickness 2
