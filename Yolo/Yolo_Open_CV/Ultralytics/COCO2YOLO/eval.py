!cd /content/yolov5;python test.py --weights /mydrive/ultra_workdir/bccd/weights/best.pt  --data /content/bccd/bccd.yaml \
                           --project /content/data/output --name=test_result --exist-ok --img 640 --iou 0.65
