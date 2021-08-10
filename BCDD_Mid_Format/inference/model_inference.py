model_ckpt = init_detector(cfg,device='cuda:0')

import pandas as pd

test_df=pd.read_csv('/content/BCCD_Dataset/BCCD/ImageSets/Main/test.txt',header=None,names=['image_id'])
test_df['img_path']='/content/BCCD_Dataset/BCCD/JPEGImages/'+test_df['image_id']+'.jpg'
test_df.head()

image=cv2.imread('/content/BCCD_Dataset/BCCD/JPEGImages/BloodImage_00350.jpg')
image,results=get_detected_img(model_ckpt,image)
plt.figure(figsize=(20,20))
plt.imshow(image)
