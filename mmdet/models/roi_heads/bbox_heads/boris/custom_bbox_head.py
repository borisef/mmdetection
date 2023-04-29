# Implementation for ROI head with weights per image

from mmdet.models.builder import HEADS, build_loss
from mmdet.models.roi_heads.bbox_heads.bbox_head import BBoxHead
from mmdet.models.roi_heads import Shared2FCBBoxHead
from mmdet.models.roi_heads.bbox_heads.convfc_bbox_head import ConvFCBBoxHead


class CustomBBoxHeadWithWeightPerImage():
    """Head With Weight Per Image
       Only implements functionality of _get_targets_single, 
       should be used with multiple inheritence before BBoxHead or other ROI head
       Any instance will use _init_ from second parent 

       """  # borisef
    def _get_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes,
                           pos_gt_labels, img_metas, cfg):
        #
        results = super()._get_target_single(pos_bboxes=pos_bboxes, neg_bboxes=neg_bboxes, pos_gt_bboxes=pos_gt_bboxes,
                                             pos_gt_labels=pos_gt_labels, img_metas=None, cfg=cfg)

        # multiply label_weights and bbox_weights if needed
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
class BBoxHeadWithWeightPerImage(CustomBBoxHeadWithWeightPerImage, BBoxHead):
    """BBoxHead With Weight Per Image
      This class simply defines multiple inheritence
      """
    pass

@HEADS.register_module()
class Shared2FCBBoxHeadWithWeightPerImage(CustomBBoxHeadWithWeightPerImage, Shared2FCBBoxHead):
    """Shared2FCBBoxHead With Weight Per Image
          This class simply defines multiple inheritence
          """
    pass

@HEADS.register_module()
class ConvFCBBoxHeadWithWeightPerImage(CustomBBoxHeadWithWeightPerImage, ConvFCBBoxHead):
    """ConvFCBBoxHead With Weight Per Image
          This class simply defines multiple inheritence
          """
    pass

#Remark: same way we can extend  Shared4Conv1FCBBoxHead,DoubleConvFCBBoxHead etc + WithWeightPerImage

