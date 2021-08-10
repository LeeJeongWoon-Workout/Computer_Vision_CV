def get_detected_img(model, img_array,  score_threshold=0.25, is_print=True):
  draw_img = img_array.copy()
  bbox_color=(0, 255, 0)
  text_color=(0, 0, 255)


  results = inference_detector(model, img_array)
  labels_to_names_seq=('RBC',"WBC",'Platelets')

  for result_ind, result in enumerate(results):
    if len(result) == 0:
      continue
    
    result_filtered = result[np.where(result[:, 4] > score_threshold)]
    
    for i in range(len(result_filtered)):
      left = int(result_filtered[i, 0])
      top = int(result_filtered[i, 1])
      right = int(result_filtered[i, 2])
      bottom = int(result_filtered[i, 3])
      caption = "{}: {:.4f}".format(labels_to_names_seq[result_ind], result_filtered[i, 4])
      cv2.rectangle(draw_img, (left, top), (right, bottom), color=bbox_color, thickness=2)
      cv2.putText(draw_img, caption, (int(left), int(top - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
      if is_print:
        print(caption)

  return draw_img,results

import matplotlib.pyplot as plt




import matplotlib.pyplot as plt
import cv2
%matplotlib inline 



def show_detected_images(model, img_arrays, ncols=5):
    figure, axs = plt.subplots(figsize=(22, 6), nrows=1, ncols=ncols)
    for i in range(ncols):
      detected_img = get_detected_img(model, img_arrays[i],  score_threshold=0.1, is_print=True)
      axs[i].imshow(detected_img)
