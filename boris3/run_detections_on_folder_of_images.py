import copy
import glob
import os.path
import os.path as osp
import sys

import cv2
import mmcv
from mmdet.registry import VISUALIZERS
from mmdet.apis import init_detector, inference_detector
import train_wrap_utils_home


config_path = "/home/borisef/projects/mm/mmdetection/boris3/config/config_faster-rcnn_r50_fpn_2x_coco.py"
ckpt_path = "/home/borisef/projects/mm/mmdetection/boris3/work_dirs/faster-rcnn_r50_fpn_2x_vehicles/epoch_24.pth"
imgs_dir = "/home/borisef/data/vehicles/test/"
out_dir = "/home/borisef/temp/out_imgs/"
to_show = True
thresh = 0.5
make_video_fps = 1 #None

if not osp.exists(out_dir):
    os.mkdir(out_dir)


imgs = train_wrap_utils_home.redefine_images(imgs_dir)

model = init_detector(config_path, ckpt_path, device='cuda:0')

visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.dataset_meta = model.dataset_meta

if(make_video_fps is not None):
    im = mmcv.imread(imgs[0])
    out_video = cv2.VideoWriter(osp.join(out_dir,'video.avi'),cv2.VideoWriter_fourcc(*'DIVX'),
                          make_video_fps,
                          (im.shape[0],im.shape[1]))

for ima in imgs:
    # im = mmcv.imread(ima)
    # result = inference_detector(model, im)
    # labels = result.pred_instances['labels'].cpu().detach().numpy()
    # bboxes = result.pred_instances['bboxes'].cpu().detach().numpy()
    # scores = result.pred_instances['scores'].cpu().detach().numpy()

    im = mmcv.imread(ima)
    result = inference_detector(model, im)
    ouf = osp.join(out_dir, osp.basename(ima))
    img = mmcv.imconvert(im, 'bgr', 'rgb')
    try:
        visualizer.add_datasample(
            'result',
            img,
            data_sample=result,
            draw_gt=False,
            wait_time=0.1,
            out_file=ouf,
            pred_score_thr=thresh,
            show=to_show)

        if (make_video_fps is not None):
            imarray = cv2.imread(ouf)
            out_video.write(imarray)
    except:
        pass

if(make_video_fps is not None):
    out_video.release()