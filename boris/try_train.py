import copy
import os.path as osp

import mmcv
import numpy as np

from mmcv import Config
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector

from  train_cfg_utils import  ArgConfigParams, update_config_from_args

cfg_file = 'configs/kiti_config.py'
#cfg_file = 'configs/car_damage_config.py'

cp = ArgConfigParams() # create inctance of class
cp.arguments_cfg() #init valid arguments in command line
args = cp.parser.parse_args()
if(args.config_file is not None):
        cfg_file = args.config_file


cfg = Config.fromfile(cfg_file)
cfg = update_config_from_args(cfg, cp) # change parameters based on command line

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

imgs = cfg.example_images
model.cfg = cfg

for ima in imgs:
    im = mmcv.imread(ima)
    result = inference_detector(model, im)
    ouf = osp.join(cfg.work_dir, 'imgs', osp.basename(ima))
    try:
        show_result_pyplot(model, im, result,score_thr=0.2 , out_file=ouf)
    except:
        pass

