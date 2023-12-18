import copy
import os.path
import os.path as osp
import sys

import cv2
import mmcv
from mmdet.registry import VISUALIZERS
from mmdet.apis import init_detector, inference_detector

import fiftyone as fo
import train_wrap_utils_home
from read_my_coco import read_my_coco # from my fiftyone library


def add_mmdete_to_dataset51(dataset,
                            config_path,
                            ckpt_path,
                            images_path = None, # if none will take from dataset
                            score_threshold = 0.5,
                            prediction_label = "mmdet_pred",
                            map_labels = None, # if None it is like  {0:0, 1:1, ...}
                            my_classes = None #if None will use dataset.default_classes
                            ):
    # Get class list
    if (my_classes is not None):
        classes = my_classes
    else:
        classes = dataset.default_classes
    # mmdetection
    # Build the model from a config file and a checkpoint file
    model = init_detector(config_path, ckpt_path, device='cuda:0')

    if(images_path is not None):
        imgs = train_wrap_utils_home.redefine_images(images_path)
    else:
        imgs = []
        for s in dataset.view():
            imgs.append(s['filepath'])

    for ima in imgs:
        im = mmcv.imread(ima)
        result = inference_detector(model, im)
        labels = result.pred_instances['labels'].cpu().detach().numpy()
        bboxes = result.pred_instances['bboxes'].cpu().detach().numpy()
        scores = result.pred_instances['scores'].cpu().detach().numpy()

        h, w, c = im.shape

        # print(result)
        sample = dataset.view()[ima]
        # Convert detections to FiftyOne format
        detections = []

        # add_predictions_to_sample(sample,labels,bboxes,scores)
        for label, score, box in zip(labels, scores, bboxes):
            # Convert to [top-left-x, top-left-y, width, height]
            # in relative coordinates in [0, 1] x [0, 1]
            x1, y1, x2, y2 = box
            rel_box = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]

            true_label = label  # mapping option
            if (map_labels is not None):
                true_label = map_labels[label]

            if score_threshold <= score and true_label < len(classes):
                detections.append(
                    fo.Detection(
                        label=classes[true_label],
                        bounding_box=rel_box,
                        confidence=score
                    )
                )

        # Save predictions to dataset
        sample[predictions_label] = fo.Detections(detections=detections)
        sample.save()

    return dataset





#flexible map labels if needed
map_labels = None # {0:0, 1:1, ...}
my_classes = None

images_path = "/home/borisef/data/vehicles/test/"
coco_json_path = "/home/borisef/data/vehicles/test/annotations.json"
config_path = "/home/borisef/projects/mm/mmdetection/boris3/work_dirs/detr_vehicles1_loadfrom/detr_r50_8xb2-150e_coco_FULL.py"
ckpt_path = "/home/borisef/projects/mm/mmdetection/boris3/work_dirs/detr_vehicles1_loadfrom/epoch_16.pth"
score_threshold = 0.5
predictions_label = "mmdet_pred"

dataset = read_my_coco(images_path,coco_json_path)
dataset.add_dynamic_sample_fields()

dataset = add_mmdete_to_dataset51(dataset,
                            config_path,
                            ckpt_path,
                            images_path = None, # if none will take from dataset
                            score_threshold = 0.5,
                            prediction_label = "mmdet_pred",
                            map_labels = None, # if None it is like  {0:0, 1:1, ...}
                            my_classes = None #if None will use dataset.default_classes
                            )

fo.launch_app(dataset)
print("OK")



