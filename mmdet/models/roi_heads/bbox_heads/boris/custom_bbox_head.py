# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from torch.nn.modules.utils import _pair

from mmdet.models.builder import HEADS, build_loss
from mmdet.models.roi_heads.bbox_heads.bbox_head import BBoxHead
from mmdet.models.roi_heads import Shared2FCBBoxHead

class BBoxHeadWithWeightPerImage():
    def _get_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes,
                           pos_gt_labels, img_metas, cfg):
        results = super()._get_target_single(pos_bboxes, neg_bboxes, pos_gt_bboxes,
                           pos_gt_labels,img_metas, cfg)


        if (img_metas is not None and img_metas is not [] and 'roi_head.loss_weight' in img_metas):
            if (img_metas['roi_head.loss_weight'] != 1):
                label_weights = results[1]
                bbox_weights = results[3]
                label_weights = label_weights * img_metas['roi_head.loss_weight']
                bbox_weights = bbox_weights * img_metas['roi_head.loss_weight']

                results = list(results)
                results[1] = label_weights
                results[3] = bbox_weights
                results = tuple(results)
        return results


@HEADS.register_module()
class BBoxHeadCustom(BBoxHeadWithWeightPerImage,BBoxHead):
    pass
@HEADS.register_module()
class Shared2FCBBoxHeadWithWeightPerImage(BBoxHeadWithWeightPerImage,Shared2FCBBoxHead):
    pass
