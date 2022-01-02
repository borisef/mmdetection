import copy
import os.path as osp

import mmcv
import numpy as np

from mmcv import Config
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector



cfg_file = 'configs/kiti_config.py'
cfg = Config.fromfile(cfg_file)

# Build dataset
datasets = [build_dataset(cfg.data.train)]

# Build the detector
model = build_detector(
    cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))

# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_detector(model, datasets, cfg, distributed=False, validate=True)


#try on one image:

img = mmcv.imread('../demo/kitti_tiny/training/image_2/000068.jpeg')

model.cfg = cfg
result = inference_detector(model, img)
show_result_pyplot(model, img, result,score_thr=0.0)#, out_file='try.jpg')