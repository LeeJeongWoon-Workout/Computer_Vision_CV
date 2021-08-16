#design
'''Yolo 프로그래밍 설계
1.cv2.dnn.readNetDarknet(config파일,weight모델)으로 pretrained 된 inference 모델 로딩
2.사용자 3개의 다른 scale 별로(multi scaling) 구성된 output layer에서 Object Detect 진행
3.사용자가 직접, NMS로 최종 결과를 필터링을 해 주어야 한다.
'''

'''---------------------------------------------------------------------------------------1 library setting'''
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


'''---------------------------------------------------------------------------------------2 model setting'''

  
#readNetFromDarknet(config파일,weight파일)을 이용하여 yolo inference network 모델을 로딩 1단계

import os
import cv2

weights_path = '/content/pretrained/yolov3.weights'
config_path =  '/content/pretrained/yolov3.cfg'
#config 파일 인자가 먼저 옴. 
#darknet으로 구성된 dnn 모델을 생성
#Feature Pyramid가 포함된 Yolo v3 모델이 생성
cv_net_yolo = cv2.dnn.readNetFromDarknet(config_path, weights_path)

labels_to_names_seq = {0:'person',1:'bicycle',2:'car',3:'motorbike',4:'aeroplane',5:'bus',6:'train',7:'truck',8:'boat',9:'traffic light',10:'fire hydrant',
                        11:'stop sign',12:'parking meter',13:'bench',14:'bird',15:'cat',16:'dog',17:'horse',18:'sheep',19:'cow',20:'elephant',
                        21:'bear',22:'zebra',23:'giraffe',24:'backpack',25:'umbrella',26:'handbag',27:'tie',28:'suitcase',29:'frisbee',30:'skis',
                        31:'snowboard',32:'sports ball',33:'kite',34:'baseball bat',35:'baseball glove',36:'skateboard',37:'surfboard',38:'tennis racket',39:'bottle',40:'wine glass',
                        41:'cup',42:'fork',43:'knife',44:'spoon',45:'bowl',46:'banana',47:'apple',48:'sandwich',49:'orange',50:'broccoli',
                        51:'carrot',52:'hot dog',53:'pizza',54:'donut',55:'cake',56:'chair',57:'sofa',58:'pottedplant',59:'bed',60:'diningtable',
                        61:'toilet',62:'tvmonitor',63:'laptop',64:'mouse',65:'remote',66:'keyboard',67:'cell phone',68:'microwave',69:'oven',70:'toaster',
                        71:'sink',72:'refrigerator',73:'book',74:'clock',75:'vase',76:'scissors',77:'teddy bear',78:'hair drier',79:'toothbrush' }


'''----------------------------------------------------------------------------------------------------------train'''

layer_names = cv_net_yolo.getLayerNames()
print('### yolo v3 layer name:', layer_names)
print('final output layer id:', cv_net_yolo.getUnconnectedOutLayers())
print('final output layer name:', [layer_names[i[0] - 1] for i in cv_net_yolo.getUnconnectedOutLayers()])
#82번 94번 106번 CNN Layer을 추출해야 한다.
#전체 Darknet layer에서 13x13 grid, 26x26, 52x52 grid에서 detect된 Output layer만 filtering
layer_names = cv_net_yolo.getLayerNames()
outlayer_names = [layer_names[i[0] - 1] for i in cv_net_yolo.getUnconnectedOutLayers()]
#CNN 넘버를 뽑기 위한 코드
print('output_layer name:', outlayer_names)

img = cv2.imread('./data/beatles01.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 로딩한 모델은 Yolov3 416 x 416 모델임. 원본 이미지 배열을 사이즈 (416, 416)으로, BGR을 RGB로 변환하여 배열 입력
# cv2.dnn.blobFromImage는 데이터 이미지를 전처리 하는 cv2 내장함수
# 그렇게 setInput으로 데이터를 집어넣는다.
cv_net_yolo.setInput(cv2.dnn.blobFromImage(img, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False))

# Object Detection 수행하여 결과를 cvOut으로 반환 
# 아까 뽑아놓은 82,94,106번 Layer에 대해서 학습을 진행한다.
cv_outs = cv_net_yolo.forward(outlayer_names)
print('cv_outs type:', type(cv_outs), 'cv_outs의 내부 원소개수:', len(cv_outs))
print(cv_outs[0].shape, cv_outs[1].shape, cv_outs[2].shape)
print(cv_outs)


'''
cv_outs type: <class 'list'> cv_outs의 내부 원소개수: 3
(507, 85) (2028, 85) (8112, 85)   -> (13*13*3,5+80) , (26*26*3,5+80) , (52*52*3,5+85)  13X13 26X26 52X52 3은 Feature Pyramid를 의미 그것을
2차원 array에 담기 위해 곱해준다. 85의 5는 중심좌표,너비,높이,Confident,80개의 COCO 클래스 softmax 결과로 구성되어 있다.

# 기억해야 할 점 13,26,52 3개의 layer을 끄집어 내고 그것에 각각 Object Detection을 취해야 한다.

[array([[0.03803749, 0.0470234 , 0.3876816 , ..., 0.        , 0.        ,
        0.        ],
       [0.04705836, 0.03385845, 0.2689603 , ..., 0.        , 0.        ,
        0.        ],
       [0.04941482, 0.03791986, 0.7151826 , ..., 0.        , 0.        ,
        0.        ],
       ...,
       [0.9585798 , 0.9460585 , 0.35046625, ..., 0.        , 0.        ,
        0.        ],
       [0.96015006, 0.9630715 , 0.29724196, ..., 0.        , 0.        ,
        0.        ],
       [0.9663636 , 0.9657401 , 0.79356086, ..., 0.        , 0.        ,
        0.        ]], dtype=float32), array([[0.01637367, 0.02457962, 0.04684627, ..., 0.        , 0.        ,
        0.        ],
       [0.01678773, 0.01458679, 0.46203217, ..., 0.        , 0.        ,
        0.        ],
       [0.02219823, 0.01376948, 0.0662718 , ..., 0.        , 0.        ,
        0.        ],
       ...,
       [0.97421783, 0.97686917, 0.04557502, ..., 0.        , 0.        ,
        0.        ],
       [0.98114103, 0.9762939 , 0.33147967, ..., 0.        , 0.        ,
        0.        ],
       [0.97884774, 0.98335934, 0.07896643, ..., 0.        , 0.        ,
        0.        ]], dtype=float32), array([[0.00859342, 0.00442324, 0.01781066, ..., 0.        , 0.        ,
        0.        ],
       [0.010101  , 0.01088366, 0.01980249, ..., 0.        , 0.        ,
        0.        ],
       [0.01071996, 0.00756924, 0.20484295, ..., 0.        , 0.        ,
        0.        ],
       ...,
       [0.9901033 , 0.9906244 , 0.01741469, ..., 0.        , 0.        ,
        0.        ],
       [0.9907341 , 0.9876037 , 0.01802968, ..., 0.        , 0.        ,
        0.        ],
       [0.98756605, 0.99131656, 0.17707303, ..., 0.        , 0.        ,
        0.        ]], dtype=float32)]
'''




'''-----------------------------------------------------------------------------------------------------------------------Object Detection'''


'''
함수 설계 : pre_trained 된 모델과 cv2로 array 처리된 이미지를 인자로 받아 최종적인 Object-Detection을 처리한다.
'''
def get_detected_img(cv_net, img_array, conf_threshold, nms_threshold, is_print=True):
  
    
    # 원본 이미지를 네트웍에 입력시에는 (416, 416)로 resize 함. 
    # 이후 결과가 출력되면 resize된 이미지 기반으로 bounding box 위치가 예측 되므로 이를 다시 원복하기 위해 원본 이미지 shape정보 필요
    rows = img_array.shape[0]
    cols = img_array.shape[1]
    
    draw_img = img_array.copy()
    
    #전체 Darknet layer에서 13x13 grid, 26x26, 52x52 grid에서 detect된 Output layer만 filtering
    layer_names = cv_net.getLayerNames()
    outlayer_names = [layer_names[i[0] - 1] for i in cv_net.getUnconnectedOutLayers()]
    
    # 로딩한 모델은 Yolov3 416 x 416 모델임. 원본 이미지 배열을 사이즈 (416, 416)으로, BGR을 RGB로 변환하여 배열 입력
    cv_net.setInput(cv2.dnn.blobFromImage(img_array, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False))
    start = time.time()
    # Object Detection 수행하여 결과를 cvOut으로 반환 
    cv_outs = cv_net.forward(outlayer_names)
    layerOutputs = cv_net.forward(outlayer_names)
    # bounding box의 테두리와 caption 글자색 지정
    green_color=(0, 255, 0)
    red_color=(0, 0, 255)

    class_ids = []
    confidences = []
    boxes = [] #[left,top,width,height]를 담는다.

    # 3개의 개별 output layer별로 Detect된 Object들에 대해서 Detection 정보 추출 및 시각화 
    for ix, output in enumerate(cv_outs):
        # Detected된 Object별 iteration
        for jx, detection in enumerate(output):
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # confidence가 지정된 conf_threshold보다 작은 값은 제외 
            if confidence > conf_threshold:
                #print('ix:', ix, 'jx:', jx, 'class_id', class_id, 'confidence:', confidence)
                # detection은 scale된 좌상단, 우하단 좌표를 반환하는 것이 아니라, detection object의 중심좌표와 너비/높이를 반환
                # 원본 이미지에 맞게 scale 적용 및 좌상단, 우하단 좌표 계산
                center_x = int(detection[0] * cols)
                center_y = int(detection[1] * rows)
                width = int(detection[2] * cols)
                height = int(detection[3] * rows)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                # 3개의 개별 output layer별로 Detect된 Object들에 대한 class id, confidence, 좌표정보를 모두 수집
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
    
    # NMS로 최종 filtering된 idxs를 이용하여 boxes, classes, confidences에서 해당하는 Object정보를 추출하고 시각화.
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    if len(idxs) > 0:
        for i in idxs.flatten():
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            # labels_to_names 딕셔너리로 class_id값을 클래스명으로 변경. opencv에서는 class_id + 1로 매핑해야함.
            caption = "{}: {:.4f}".format(labels_to_names_seq[class_ids[i]], confidences[i])
            #cv2.rectangle()은 인자로 들어온 draw_img에 사각형을 그림. 위치 인자는 반드시 정수형.
            cv2.rectangle(draw_img, (int(left), int(top)), (int(left+width), int(top+height)), color=green_color, thickness=2)
            cv2.putText(draw_img, caption, (int(left), int(top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red_color, 1)

    if is_print:
        print('Detection 수행시간:',round(time.time() - start, 2),"초")
    return draw_img
  
  '''NMS 수행 로직
  1. Detected 된 bounding box별로 특정 Confidence threshold 이하
    bounding box는 먼저 제거(confidence score < 0.5)
  2. 가장 높은 confidence score를 가진 box 순으로 내림차순 정렬하고
    아래 로직을 모든 box에 순차적으로 적용.
    • 높은 confidence score를 가진 box와 겹치는 다른 box를 모두
    조사하여 IOU가 특정 threshold 이상인 box를 모두 제거(예: 
    IOU Threshold > 0.4 )
   3. 남아 있는 box만 선택
  
  >>
  nms_threshold는 confidence가 가장 높은 bbox와 겹치는 정도를 나타내는데 많이 겹칠 수록 같은 사물을 가리키고 있기 때문에 없애는 것이 좋다.
  이 값이 낮을 수록 많은 박스들이 사라질 것
  '''
  
  
  
  '''------------------------------------------------------------------------------------------------------------------------------------Inference'''
  import cv2
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import time
import os

# image 로드 
img = cv2.imread('/content/data/beatles01.jpg')

weights_path = '/content/pretrained/yolov3.weights'
config_path =  '/content/pretrained/yolov3.cfg'

# darknet yolo pretrained 모델 로딩
cv_net_yolo = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    
conf_threshold = 0.5
nms_threshold = 0.4
# Object Detetion 수행 후 시각화 
draw_img = get_detected_img(cv_net_yolo, img, conf_threshold=conf_threshold, nms_threshold=nms_threshold, is_print=True)

img_rgb = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 12))
plt.imshow(img_rgb)

conf_threshold = 0.5
nms_threshold = 0.4
# Object Detetion 수행 후 시각화 
draw_img = get_detected_img(cv_net_yolo, img, conf_threshold=conf_threshold, nms_threshold=nms_threshold, is_print=True)

img_rgb = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 12))
plt.imshow(img_rgb)
