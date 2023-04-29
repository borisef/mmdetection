#<RFL> Similar to  test_bbox_head_loss, some changes applied to test custom bbox head with weights
import mmcv
import torch

from mmdet.core import bbox2roi
try:
    from ..utils import _dummy_bbox_sampling
except:
    import sys
    from os.path import dirname

    sys.path.insert(0, dirname(dirname(dirname(dirname(__file__)))))  # go up to "tests"
    from test_models.test_roi_heads.utils import _dummy_bbox_sampling

from mmdet.models.roi_heads.bbox_heads.boris.custom_bbox_head import ConvFCBBoxHeadWithWeightPerImage,Shared2FCBBoxHeadWithWeightPerImage, BBoxHeadWithWeightPerImage

def _custom_bbox_head_loss(bbox_class=BBoxHeadWithWeightPerImage):
    """Tests bbox head loss when truth is empty and non-empty.
    This function can be used with any tested class
    First we test legacy case (without weights) identical test as in test_bbox_head_loss.py
    At the end we test with weight
    """
    if(bbox_class is BBoxHeadWithWeightPerImage):
        self = bbox_class(in_channels=8, roi_feat_size=3)
    elif(bbox_class is Shared2FCBBoxHeadWithWeightPerImage):
        self = Shared2FCBBoxHeadWithWeightPerImage(in_channels=8, roi_feat_size=3)
    elif(bbox_class is ConvFCBBoxHeadWithWeightPerImage):
        self = ConvFCBBoxHeadWithWeightPerImage(in_channels=8, roi_feat_size=3,num_shared_fcs=2)
    else:
        raise NotImplementedError("Not supported bbox_class" + str(bbox_class))

    # Dummy proposals
    proposal_list = [
        torch.Tensor([[23.6667, 23.8757, 228.6326, 153.8874]]),
    ]

    target_cfg = mmcv.Config(dict(pos_weight=1))

    # Test bbox loss when truth is empty
    gt_bboxes = [torch.empty((0, 4))]
    gt_labels = [torch.LongTensor([])]

    sampling_results = _dummy_bbox_sampling(proposal_list, gt_bboxes,
                                            gt_labels)

    bbox_targets = self.get_targets(sampling_results, gt_bboxes, gt_labels,
                                    target_cfg)#add img_metas as last input
    labels, label_weights, bbox_targets, bbox_weights = bbox_targets

    # Create dummy features "extracted" for each sampled bbox
    num_sampled = sum(len(res.bboxes) for res in sampling_results)
    rois = bbox2roi([res.bboxes for res in sampling_results])
    dummy_feats = torch.rand(num_sampled, 8 * 3 * 3)
    cls_scores, bbox_preds = self.forward(dummy_feats)

    losses = self.loss(cls_scores, bbox_preds, rois, labels, label_weights,
                       bbox_targets, bbox_weights)
    assert losses.get('loss_cls', 0) > 0, 'cls-loss should be non-zero'
    assert losses.get('loss_bbox', 0) == 0, 'empty gt loss should be zero'

    # Test bbox loss when truth is non-empty
    gt_bboxes = [
        torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
    ]
    gt_labels = [torch.LongTensor([2])]

    sampling_results = _dummy_bbox_sampling(proposal_list, gt_bboxes,
                                            gt_labels)
    rois = bbox2roi([res.bboxes for res in sampling_results])

    bbox_targets = self.get_targets(sampling_results, gt_bboxes, gt_labels,
                                    target_cfg)
    labels, label_weights, bbox_targets, bbox_weights = bbox_targets

    # Create dummy features "extracted" for each sampled bbox
    num_sampled = sum(len(res.bboxes) for res in sampling_results)
    dummy_feats = torch.rand(num_sampled, 8 * 3 * 3)
    cls_scores, bbox_preds = self.forward(dummy_feats)

    losses = self.loss(cls_scores, bbox_preds, rois, labels, label_weights,
                       bbox_targets, bbox_weights)
    assert losses.get('loss_cls', 0) > 0, 'cls-loss should be non-zero'
    assert losses.get('loss_bbox', 0) > 0, 'box-loss should be non-zero'

    #last test with weights
    img_metas_1 = [{
        'roi_head.loss_weight': 100
    }]
    bbox_targets_with_weights = self.get_targets(sampling_results, gt_bboxes, gt_labels,
                                    target_cfg, img_metas=img_metas_1)
    labels_w, label_weights_w, bbox_targets_w, bbox_weights_w = bbox_targets_with_weights
    losses_w = self.loss(cls_scores, bbox_preds, rois, labels_w, label_weights_w,
                       bbox_targets_w, bbox_weights_w)
    assert losses_w.get('loss_cls', 0) > losses.get('loss_cls', 0), 'cls-loss should be 100 larger than without weight'
    assert losses_w.get('loss_bbox', 0) > losses.get('loss_bbox', 0), 'cls-loss should be 100 larger than without weight'
    pass

def test_custom_bbox_head_loss():
    #test all cutsom bbox heads with weight per image
    _custom_bbox_head_loss(bbox_class=BBoxHeadWithWeightPerImage)
    _custom_bbox_head_loss(bbox_class=Shared2FCBBoxHeadWithWeightPerImage)
    _custom_bbox_head_loss(bbox_class=ConvFCBBoxHeadWithWeightPerImage)


if __name__=="__main__":
    test_custom_bbox_head_loss()