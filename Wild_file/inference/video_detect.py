#pretrained 되고 eval가 완료된 모델의 config과 weights를 실제 inference하기 위해서 model.cfg=cfg로 초기화를 다시 반드시 해야 한다.
model.cfg=cfg
do_detected_video(model,'/content/data/smoke.mp4','/content/data/smoke_out.mp4')
