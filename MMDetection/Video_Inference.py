!wget -O /content/data/John_Wick_small.mp4 https://github.com/chulminkw/DLCV/blob/master/data/video/John_Wick_small.mp4?raw=true
  
from mmdet.apis import init_detector, inference_detector
import mmcv

config_file = '/content/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = '/content/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')


video_reader = mmcv.VideoReader('/content/data/John_Wick_small.mp4')
video_writer = None
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter('/content/data/John_Wick_small_out1.mp4', fourcc, video_reader.fps,(video_reader.width, video_reader.height))
#결과물 출력 주소,동영상 형식,fps(mmcv.VideoReader의 내장함수에 fps를 구하는 함수가 있다),(너비,높이))
for frame in mmcv.track_iter_progress(video_reader): #프레임이 없을 때까지 진행한다.
  result = inference_detector(model, frame) #모델에 이미지를 삽입해서 프레임별로 Object_Detection을 수행한다.
  frame = model.show_result(frame, result, score_thr=0.4)
#frame의 결과에서 0.4 이상의 Object_Detection만 frame에 표시한다.
  video_writer.write(frame)
#결과를 video_writer에 write한다.
if video_writer:
        video_writer.release() #출력한다.
