# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.ops import batched_nms

from mmdet.models.builder import HEADS
from mmdet.models.dense_heads.rpn_head import RPNHead

#<RFL>


class HeadWithWeightPerImage():
    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            label_channels=1,
                            unmap_outputs=True):
        #call default _get_targets_single from parent of the class (RPNHead in our case)
        results = super()._get_targets_single( flat_anchors,
                                                                   valid_flags,
                                                                   gt_bboxes,
                                                                   gt_bboxes_ignore,
                                                                   gt_labels,
                                                                   img_meta,
                                                                   label_channels)

        #(labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds, sampling_result) = results
        # multiply label_weights and bbox_weights
        if('rpn_head.loss_weight' in img_meta):
            if (img_meta['rpn_head.loss_weight'] != 1):
                label_weights = results[1]
                bbox_weights = results[3]
                label_weights = label_weights * img_meta['rpn_head.loss_weight']
                bbox_weights = bbox_weights * img_meta['rpn_head.loss_weight']
                results = list(results)
                results[1] = label_weights
                results[3] = bbox_weights
                results = tuple(results)
        return results

@HEADS.register_module()
class RPNHeadWithWeightPerImage(HeadWithWeightPerImage,RPNHead):
    pass
#REMARK: Only RPNHead with such capability is implemented for now, but any other head inherited from AnchorHead can be easily updated too
# for example GuidedAnchorHead


