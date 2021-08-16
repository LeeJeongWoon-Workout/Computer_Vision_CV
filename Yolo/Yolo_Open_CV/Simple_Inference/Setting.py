'''Yolo 프로그래밍 설계
1.cv2.dnn.readNetDarknet(config파일,weight모델)으로 pretrained 된 inference 모델 로딩
2.사용자 3개의 다른 scale 별로(multi scaling) 구성된 output layer에서 Object Detect 진행
3.사용자가 직접, NMS로 최종 결과를 필터링을 해 주어야 한다.
'''
#입력 이미지로 사용될 이미지 다운로드 하기
#이미지를 담을 data directory 생성

!mkdir /content/data
!wget -O ./data/beatles01.jpg https://raw.githubusercontent.com/chulminkw/DLCV/master/data/image/beatles01.jpg
  
### coco 데이터 세트로 pretrained 된 yolo weight 파일과 config 파일 다운로드하여 /content/pretrained 디렉토리 아래에 저장. 
!mkdir ./pretrained
!echo "##### downloading pretrained yolo/tiny-yolo weight file and config file"
!wget -O /content/pretrained/yolov3.weights https://pjreddie.com/media/files/yolov3.weights
!wget -O /content/pretrained/yolov3.cfg https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg?raw=true 

!wget -O /content/pretrained/yolov3-tiny.weights https://pjreddie.com/media/files/yolov3-tiny.weights
!wget -O /content/pretrained/yolov3-tiny.cfg https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg?raw=true

!ls /content/pretrained
