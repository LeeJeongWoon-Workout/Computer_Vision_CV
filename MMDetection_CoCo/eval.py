import torch
dataset = build_dataset(cfg.data.test)
#파이토치 기반 데이터 로더 생성
#MMdetection을 eval하기 위해서는 MDataParallel로 만든 평가용 포델과,파이토치 데이터 로더를
#single_gpu_test에 집어 넣어야 한다.
'''
설계
1.data loader를 생성한다. (gpu값을 모두 1로 초기화 해야 single_gpu_test가 작동한다.)
2.MMDataParallel를 통해 eval용 model를 하나 더 만든다.
3.single_gpu_test(model,data_loader,True,~저장위치)를 집어 넎느다.
'''
data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1, #single_gpu_test를 사용하기 위해 gpu 값을 1로 초기화 해야 한다.
        workers_per_gpu=1,
        dist=False,
        shuffle=False)
#MMDetection Evaluation을 위한 함수    
model_ckpt = MMDataParallel(model, device_ids=[0])
#결과물
outputs = single_gpu_test(model_ckpt, data_loader, True, '/content/tutorial_exps', 0.3)

print('결과 outputs type:', type(outputs))
print('evalution 된 파일의 갯수:', len(outputs))
print('첫번째 evalutation 결과의 type:', type(outputs[0]))
print('첫번째 evaluation 결과의 CLASS 갯수:', len(outputs[0]))
print('첫번째 evaluation 결과의 CLASS ID 0의 type과 shape', type(outputs[0][0]), outputs[0][0].shape)

'''
1. 파일 이름을 os.listdir 로 한번에 저장한다.
2. file_path를 만든다.
3. img_array를 생성한다.
'''

path_dir='/content/tutorial_exps'
file_list = os.listdir(path_dir)
file_path=['/content/tutorial_exps/'+x for x in file_list]

img_array=[cv2.imread(x) for x in file_path]


#이미지 출력 함수 subplots 사용
def show_detected_images(img_arrays, ncols=50):
    figure, axs = plt.subplots(figsize=(22, 6), nrows=1, ncols=ncols)
    for i in range(ncols):
      axs[i].imshow(img_arrays[i])

for i in range(50):
  show_detected_images(img_array[i:5*(i+1)], ncols=5)
