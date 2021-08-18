'''-------------------------------------------------------------------------------------Setting'''
!git clone https://github.com/ultralytics/yolov3
!cd yolov3;pip install -qr requirements.txt

import torch
from IPython.display import Image, clear_output  # to display images

clear_output()
print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

'''-----------------------------------------------------------------------------------train'''
%cd yolov3
!python train.py --img 640 --batch 16 --epochs 3 --data coco128.yaml --weights yolov3.pt --nosave 
'''
train.py의 data option값으로 Dataset config yaml 파일을 지정할 수 있으며, 파일명만 입력할 경우는 yolov3/data 디렉토리 아래에서 해당 파일을 찾음. 
절대 경로로 입력할 경우 해당 경로에서 찾음.
weights option의 경우 파일명만 입력할 경우 yolov3 디렉토리에서 해당 파일을 찾음. 해당 파일이 없을 경우 자동으로 해당 파일을 https://github.com/ultralytics/yolov3/releases 에서 Download 함. 절대 경로를 입력한 경우 해당 경로에서 파일을 찾되 파일이 없으면 해당 경로로 자동 Download함.
weights 파일은 yolov3.pt, yolov3-tiny.pt, yolov3-spp.pt
'''

'''------------------------------------------------------------------------------------Detection Inference 수행'''

!cd yolov3;python detect.py --weights yolov3.pt --img 640 --conf 0.25 --source /content/data/Night_Day_Chase.mp4 \
                            --project=/content/data/run_video --exist-ok --line-thickness 1

'''weights-? pretrained model,
   conf-> confidence          
   source-> image or video 
   project-> output_restorate address
   name-> file name
   exist-> if exist do not make other file
   '''
