# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet.core.bbox.assigners import MaxIoUAssigner
from mmdet.core.bbox.samplers import (OHEMSampler, RandomSampler,
                                      ScoreHLRSampler)

from mmdet.core.bbox.samplers.boris.custom_sampler import RandomSamplerWithIgnore, OHEMSamplerWithIgnore

#see test_sampler.py
# we use same code: test_ohem_sampler, test_random_sampler but with ignore negatives


def _context_for_ohem():
    import sys
    from os.path import dirname
    sys.path.insert(0, dirname(dirname(dirname(dirname(__file__))))) # go up to "tests"
    from test_models.test_forward import _get_detector_cfg

    model = _get_detector_cfg(
        'faster_rcnn/faster_rcnn_r50_fpn_ohem_1x_coco.py')
    model['pretrained'] = None

    from mmdet.models import build_detector
    context = build_detector(model).roi_head
    return context


def test_random_sampler_with_ignore_neg():
    assigner = MaxIoUAssigner(
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        ignore_iof_thr=0.5,
        ignore_wrt_candidates=False,
    )
    bboxes = torch.FloatTensor([
        [0, 0, 10, 10],
        [10, 10, 20, 20],
        [5, 5, 15, 15],
        [32, 32, 38, 42],
    ])
    gt_bboxes = torch.FloatTensor([
        [0, 0, 10, 9],
        [0, 10, 10, 19],
    ])
    gt_labels = torch.LongTensor([1, 2])
    gt_bboxes_ignore = torch.Tensor([
        [30, 30, 40, 40],
    ])
    assign_result1 = assigner.assign(
        bboxes,
        gt_bboxes,
        gt_bboxes_ignore=gt_bboxes_ignore,
        gt_labels=gt_labels)

    assign_result2 = assigner.assign(
        bboxes,
        gt_bboxes,
        gt_bboxes_ignore=gt_bboxes_ignore,
        gt_labels=gt_labels)

    assign_result = assigner.assign(
        bboxes,
        gt_bboxes,
        gt_bboxes_ignore=gt_bboxes_ignore,
        gt_labels=gt_labels)


    sampler = RandomSamplerWithIgnore(
        num=10, pos_fraction=0.5, neg_pos_ub=-1, add_gt_as_proposals=True)

    # with ignore = True
    img_metas = dict(ignore_negatives=True)
    sample_result = sampler.sample(assign_result, bboxes, gt_bboxes, gt_labels, img_metas = img_metas)

    assert len(sample_result.pos_bboxes) == len(sample_result.pos_inds)
    assert len(sample_result.neg_bboxes) == len(sample_result.neg_inds)
    assert len(sample_result.neg_inds)==0

    # with ignore = False
    img_metas = dict(ignore_negatives=False)
    sample_result = sampler.sample(assign_result1, bboxes, gt_bboxes, gt_labels, img_metas=img_metas)
    assert len(sample_result.neg_inds) > 0

    # without img_metas argument
    sample_result = sampler.sample(assign_result2, bboxes, gt_bboxes, gt_labels)
    assert len(sample_result.neg_inds) > 0


def test_ohem_sampler_with_ignore_neg():

    assigner = MaxIoUAssigner(
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        ignore_iof_thr=0.5,
        ignore_wrt_candidates=False,
    )
    bboxes = torch.FloatTensor([
        [0, 0, 10, 10],
        [10, 10, 20, 20],
        [5, 5, 15, 15],
        [32, 32, 38, 42],
    ])
    gt_bboxes = torch.FloatTensor([
        [0, 0, 10, 9],
        [0, 10, 10, 19],
    ])
    gt_labels = torch.LongTensor([1, 2])
    gt_bboxes_ignore = torch.Tensor([
        [30, 30, 40, 40],
    ])
    assign_result = assigner.assign(
        bboxes,
        gt_bboxes,
        gt_bboxes_ignore=gt_bboxes_ignore,
        gt_labels=gt_labels)

    context = _context_for_ohem()

    sampler = OHEMSamplerWithIgnore(
        num=10,
        pos_fraction=0.5,
        context=context,
        neg_pos_ub=-1,
        add_gt_as_proposals=True)

    # with ignore = True
    img_metas = dict(ignore_negatives=True)


    feats = [torch.rand(1, 256, int(2**i), int(2**i)) for i in [6, 5, 4, 3, 2]]
    sample_result = sampler.sample(
        assign_result, bboxes, gt_bboxes, gt_labels, feats=feats,img_metas=img_metas)

    assert len(sample_result.pos_bboxes) == len(sample_result.pos_inds)
    assert len(sample_result.neg_bboxes) == len(sample_result.neg_inds)
    assert len(sample_result.neg_inds)==0

if __name__ == "__main__":

    test_random_sampler_with_ignore_neg()
    test_ohem_sampler_with_ignore_neg()