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
