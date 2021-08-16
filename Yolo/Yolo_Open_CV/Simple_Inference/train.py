layer_names = cv_net_yolo.getLayerNames()
print('### yolo v3 layer name:', layer_names)
print('final output layer id:', cv_net_yolo.getUnconnectedOutLayers())
print('final output layer name:', [layer_names[i[0] - 1] for i in cv_net_yolo.getUnconnectedOutLayers()])

#전체 Darknet layer에서 13x13 grid, 26x26, 52x52 grid에서 detect된 Output layer만 filtering
layer_names = cv_net_yolo.getLayerNames()
outlayer_names = [layer_names[i[0] - 1] for i in cv_net_yolo.getUnconnectedOutLayers()]
#CNN 넘버를 뽑기 위한 코드
print('output_layer name:', outlayer_names)

img = cv2.imread('./data/beatles01.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 로딩한 모델은 Yolov3 416 x 416 모델임. 원본 이미지 배열을 사이즈 (416, 416)으로, BGR을 RGB로 변환하여 배열 입력
cv_net_yolo.setInput(cv2.dnn.blobFromImage(img, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False))

# Object Detection 수행하여 결과를 cvOut으로 반환 
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
