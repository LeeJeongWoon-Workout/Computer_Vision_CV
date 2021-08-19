# Run YOLOv3 on COCO val2017
!cd yolov3; python test.py --weights /mydrive/ultra_workdir/pet/weights/best.pt  --data /content/ox_pet/ox_pet.yaml \
                           --project /content/data/output --name=test_result --exist-ok --img 640 --iou 0.65

Image(filename='/content/data/output/test_result/confusion_matrix.png', width=800)       
