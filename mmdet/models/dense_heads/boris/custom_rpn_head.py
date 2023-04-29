# Implementation for RPN head with weights per image
from mmdet.models.builder import HEADS
from mmdet.models.dense_heads.rpn_head import RPNHead

class HeadWithWeightPerImage():
    """Head With Weight Per Image
    Only implements functionality of _get_targets_single, 
    should be used with multiple inheritence before RPNHead or AnchorHead or other head
    Any instance will use _init_ from second parent 
        
    """  # borisef
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
                                               label_channels,
                                               unmap_outputs)

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
    """RPNHead With Weight Per Image
    This class simply defines multiple inheritence
    """
    pass
#REMARK: Only RPNHead with such capability is implemented for now, but any other head inherited from AnchorHead can be easily updated too
# for example GuidedAnchorHead


