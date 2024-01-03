import os.path as osp
import sys

from mmengine.config import Config

import mmcv
from mmdet.registry import VISUALIZERS
from mmdet.apis import init_detector, inference_detector


import train_wrap_utils_home

import tools.train as train


#Faster
#config_file = "/home/borisef/projects/mm/mmdetection/boris3/config/faster-rcnn_r50_fpn_1x_coco_FULL.py" # works
#config_file = "/home/borisef/projects/mm/mmdetection/boris3/config/faster-rcnn_r50-caffe_fpn_ms-1x_coco_FULL.py" #works
#config_file = "/home/borisef/projects/mm/mmdetection/boris3/config/faster-rcnn_r50_fpn_1x_vehicles_FULL.py"#works
#config_file = "/home/borisef/projects/mm/mmdetection/boris3/config/faster-rcnn_r18_fpn_1x_vehicles_FULL.py"#works
config_file = "/home/borisef/projects/mm/mmdetection/boris3/config/config_faster-rcnn_r50_fpn_2x_coco.py"

##co-detr from projects
sys.argv.append(config_file)




train.main() #RUN TRAIN

cfg = Config.fromfile(config_file)

work_dir = cfg.work_dir

#get last_checkpoint or epoch_XX.pth with latest XX
last_checkpoint = train_wrap_utils_home.last_chkp(work_dir)

# Build the model from a config file and a checkpoint file
model = init_detector(config_file, last_checkpoint, device='cuda:0')

#try on one image:
# Init visualizer
visualizer = VISUALIZERS.build(model.cfg.visualizer)
# The dataset_meta is loaded from the checkpoint and
# then pass to the model in init_detector
visualizer.dataset_meta = model.dataset_meta

# Test a single image and show the results
img = '/home/borisef/data/vehicles/test/259ff749ac781352_jpg.rf.CZyKDHyPjIcTpIwNJ2rd.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)

# Show the results
img = mmcv.imread(img)
img = mmcv.imconvert(img, 'bgr', 'rgb')


visualizer.add_datasample(
    'result',
    img,
    data_sample=result,
    draw_gt=False,
    wait_time = 0.5,
    out_file = None,
    pred_score_thr = 0.3,
    show=True)

imgs = cfg.example_images

imgs = train_wrap_utils_home.redefine_images(imgs)

for ima in imgs:
    im = mmcv.imread(ima)
    result = inference_detector(model, im)
    ouf = osp.join(cfg.work_dir, 'imgs', osp.basename(ima))
    img = mmcv.imconvert(im, 'bgr', 'rgb')
    try:
        visualizer.add_datasample(
            'result',
            img,
            data_sample=result,
            draw_gt=False,
            wait_time=0.1,
            out_file=ouf,
            pred_score_thr=0.5,
            show=True)
    except:
        pass

print("OK")