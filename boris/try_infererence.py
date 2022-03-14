from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import matplotlib
matplotlib.use('TkAgg')
config_file = '/home/borisef/projects/mmdet/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = '/home/borisef/projects/mmdet/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
device = 'cuda:0'
# init a detector
model = init_detector(config_file, checkpoint_file, device=device)
# inference the demo image
img = '/home/borisef/projects/mmdet/mmdetection/demo/demo.jpg'
results = inference_detector(model, img)
print(results)
# show the results, may be need sudo apt-get install python3-tk

show_result_pyplot(model, img, results, score_thr = 0.5)