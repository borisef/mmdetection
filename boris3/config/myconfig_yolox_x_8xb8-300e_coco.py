_base_ = '/home/borisef/projects/mm/mmdetection/boris3/config/myconfig_yolox_s_8xb8-300e_coco.py'

# model settings
model = dict(
    backbone=dict(deepen_factor=1.33, widen_factor=1.25),
    neck=dict(
        in_channels=[320, 640, 1280], out_channels=320, num_csp_blocks=4),
    bbox_head=dict(in_channels=320, feat_channels=320))




log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = "/home/borisef/projects/mm/mmdetection/checkpoints/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth"
resume = False
launcher = 'none'
work_dir = './work_dirs/yolox_vehicles'


