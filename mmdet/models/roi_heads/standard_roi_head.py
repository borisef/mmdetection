# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin


#TODO:B: move out
from torch.autograd import Function
import numpy as np

# Autograd Function objects are what record operation history on tensors,
# and define formulas for the forward and backprop.

class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        # Store context for backprop
        ctx.alpha = alpha

        # Forward pass is a no-op
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Backward pass is just to -alpha the gradient
        output = grad_output.neg() * ctx.alpha

        # Must return same number as inputs to forward()
        return output, None



@HEADS.register_module()
class StandardRoIHead(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """Simplest base roi head including one bbox head and one mask head."""

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'], )
        return outs

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)
        self.sampling_results = sampling_results #B: keep for future use
        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        return losses

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    #B
    def probe_keep_debug_results_bbox_forward_train(self, bbox_results, bbox_targets):
        if (hasattr(self, 'keep_debug_results')):
            if (self.keep_debug_results):
                self.debug_results = {}
                self.debug_results['features'] = bbox_results['bbox_feats'].clone().cpu().detach().numpy()
                self.debug_results['labels'] = bbox_targets[0].clone().cpu().detach().numpy()

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)

        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)
        #B: TODO: make one line
        self.probe_keep_debug_results_bbox_forward_train(bbox_results, bbox_targets)


        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w).
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score).
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale the results to
                the original image. Default: True.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
                                                      proposal_list,
                                                      self.test_cfg)
        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(x, img_metas, det_bboxes,
                                              det_labels)
            return [(bbox_results, segm_results)]
        else:
            return [bbox_results]

    def onnx_export(self, x, proposals, img_metas, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        det_bboxes, det_labels = self.bbox_onnx_export(
            x, img_metas, proposals, self.test_cfg, rescale=rescale)

        if not self.with_mask:
            return det_bboxes, det_labels
        else:
            segm_results = self.mask_onnx_export(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return det_bboxes, det_labels, segm_results

    def mask_onnx_export(self, x, img_metas, det_bboxes, det_labels, **kwargs):
        """Export mask branch to onnx which supports batch inference.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            det_bboxes (Tensor): Bboxes and corresponding scores.
                has shape [N, num_bboxes, 5].
            det_labels (Tensor): class labels of
                shape [N, num_bboxes].

        Returns:
            Tensor: The segmentation results of shape [N, num_bboxes,
                image_height, image_width].
        """
        # image shapes of images in the batch

        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
            raise RuntimeError('[ONNX Error] Can not record MaskHead '
                               'as it has not been executed this time')
        batch_size = det_bboxes.size(0)
        # if det_bboxes is rescaled to the original image size, we need to
        # rescale it back to the testing scale to obtain RoIs.
        det_bboxes = det_bboxes[..., :4]
        batch_index = torch.arange(
            det_bboxes.size(0), device=det_bboxes.device).float().view(
                -1, 1, 1).expand(det_bboxes.size(0), det_bboxes.size(1), 1)
        mask_rois = torch.cat([batch_index, det_bboxes], dim=-1)
        mask_rois = mask_rois.view(-1, 5)
        mask_results = self._mask_forward(x, mask_rois)
        mask_pred = mask_results['mask_pred']
        max_shape = img_metas[0]['img_shape_for_onnx']
        num_det = det_bboxes.shape[1]
        det_bboxes = det_bboxes.reshape(-1, 4)
        det_labels = det_labels.reshape(-1)
        segm_results = self.mask_head.onnx_export(mask_pred, det_bboxes,
                                                  det_labels, self.test_cfg,
                                                  max_shape)
        segm_results = segm_results.reshape(batch_size, num_det, max_shape[0],
                                            max_shape[1])
        return segm_results

    def bbox_onnx_export(self, x, img_metas, proposals, rcnn_test_cfg,
                         **kwargs):
        """Export bbox branch to onnx which supports batch inference.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (Tensor): Region proposals with
                batch dimension, has shape [N, num_bboxes, 5].
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.

        Returns:
            tuple[Tensor, Tensor]: bboxes of shape [N, num_bboxes, 5]
                and class labels of shape [N, num_bboxes].
        """
        # get origin input shape to support onnx dynamic input shape
        assert len(
            img_metas
        ) == 1, 'Only support one input image while in exporting to ONNX'
        img_shapes = img_metas[0]['img_shape_for_onnx']

        rois = proposals

        batch_index = torch.arange(
            rois.size(0), device=rois.device).float().view(-1, 1, 1).expand(
                rois.size(0), rois.size(1), 1)

        rois = torch.cat([batch_index, rois[..., :4]], dim=-1)
        batch_size = rois.shape[0]
        num_proposals_per_img = rois.shape[1]

        # Eliminate the batch dimension
        rois = rois.view(-1, 5)
        bbox_results = self._bbox_forward(x, rois)
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']

        # Recover the batch dimension
        rois = rois.reshape(batch_size, num_proposals_per_img, rois.size(-1))
        cls_score = cls_score.reshape(batch_size, num_proposals_per_img,
                                      cls_score.size(-1))

        bbox_pred = bbox_pred.reshape(batch_size, num_proposals_per_img,
                                      bbox_pred.size(-1))
        det_bboxes, det_labels = self.bbox_head.onnx_export(
            rois, cls_score, bbox_pred, img_shapes, cfg=rcnn_test_cfg)

        return det_bboxes, det_labels
#B:
@HEADS.register_module()
class StandardRoIHeadWithExtraBBoxHead(StandardRoIHead):
    """Same  as StandardRoIHead with extra BBoxHead"""

    def __init__(self,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 extra_bbox_head=None,
                 extra_head_params=None,
                 extra_head_with_grad_reversal = False, #should insert grad reversal?
                 extra_head_temprature_params = None,#weight with respect to epoch/iter
                 extra_head_image_instance_weight=[1, 1],#weight some samples differently (NOT IMPLEMENTED)
                 extra_head_annotation_per_image = True,
                 extra_head_lambda_params=dict(max_epochs=100, #lambda will reach max at max_epoch
                                               iters_per_epoch=1000,# for internal update of epoch
                                               power_factor=3.0,# in (0, 10) larger factor faster the lambda grows up
                                               default_lambda=None, #replaces all
                                               starting_epoch = 0),#lambda is 0 till the starting_epoch
                 extra_label = None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):

        super(StandardRoIHeadWithExtraBBoxHead, self).__init__( bbox_roi_extractor=bbox_roi_extractor,
                 bbox_head=bbox_head,
                 mask_roi_extractor=mask_roi_extractor,
                 mask_head=mask_head,
                 shared_head=shared_head,
                 train_cfg=train_cfg,
                 test_cfg=test_cfg,
                 pretrained=pretrained,
                 init_cfg=init_cfg)
        #build extra head
        if extra_bbox_head is not None:
            self.init_extra_bbox_head(bbox_roi_extractor, extra_bbox_head)
            self.with_extra_bbox = True
        else:
            self.with_extra_bbox = False


        #self.extra_bbox_head = build_head(extra_bbox_head)
        self.extra_head_with_grad_reversal = extra_head_with_grad_reversal
        self.extra_head_temprature_params = extra_head_temprature_params
        self.extra_head_image_instance_weight = extra_head_image_instance_weight
        self.extra_head_annotation_per_image = extra_head_annotation_per_image
        self.extra_label = extra_label
        self.extra_head_lambda_params = extra_head_lambda_params
        if(extra_head_lambda_params is not None):
            self.extra_head_lambda_params.curr_epoch = 0
            self.extra_head_lambda_params.curr_iter = 0
            self.extra_head_lambda_params.internal_epoch_counter = True #do I count epochs by myself


    def init_extra_bbox_head(self, bbox_roi_extractor, extra_bbox_head):
        """Initialize ``bbox_head``"""
        self.extra_bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)#TODO: B: may be can use bbox_roi_extractor
        self.extra_bbox_head = build_head(extra_bbox_head)
    def calc_grl_lambda(self):
        grl_lambda = 1.0  # B
        if (self.extra_head_lambda_params is not None):
            if (self.extra_head_lambda_params['internal_epoch_counter']):
                self.extra_head_lambda_params['curr_iter'] = self.extra_head_lambda_params['curr_iter'] + 1
                self.extra_head_lambda_params['curr_epoch'] = self.extra_head_lambda_params['curr_iter'] / \
                                                              self.extra_head_lambda_params['iters_per_epoch']
            if (self.extra_head_lambda_params['default_lambda'] is not None):
                grl_lambda = self.extra_head_lambda_params['default_lambda']
            else:
                if (self.extra_head_lambda_params['curr_epoch'] < self.extra_head_lambda_params['starting_epoch']):
                    grl_lambda = 0.0
                else:
                    p = float(self.extra_head_lambda_params['curr_epoch']) / self.extra_head_lambda_params['max_epochs']
                    p, power = min(p, 1.0), self.extra_head_lambda_params['power_factor']
                    grl_lambda = 2. / (1. + np.exp(-power * p)) - 1
        return grl_lambda

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):

        #B: do StandardRoIHead forward_train first
        roi_losses = super(StandardRoIHeadWithExtraBBoxHead, self).forward_train(x=x,
                                                                    img_metas=img_metas,
                                                                    proposal_list=proposal_list,
                                                                    gt_bboxes=gt_bboxes,
                                                                    gt_labels=gt_labels,
                                                                    gt_bboxes_ignore=gt_bboxes_ignore,
                                                                    gt_masks = gt_masks,
                                                                    **kwargs)

        #TODO: forward train extra_head get loss and append it to roi_losses
        # assign gts and sample proposals
        # B: it was alreday done in father's forward, re-use sampling_results
        sampling_results = self.sampling_results
        #TODO: B: sampling results contain wrong labels (need to be replaced with extra_labels somehow)

        # bbox head forward and loss
        if self.with_extra_bbox:

            gt_extra_labels = kwargs['gt_extra_labels'] #B:take care of labels
            extra_bbox_results = self._extra_bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_extra_labels,
                                                    img_metas)
            #change names in extra_bbox_results['loss_extra_bbox'] to have all losses
            extra_bbox_results['loss_extra_bbox']['loss_extra_cls'] = extra_bbox_results['loss_extra_bbox']['loss_cls']
            extra_bbox_results['loss_extra_bbox']['loss_extra_bbox'] = extra_bbox_results['loss_extra_bbox']['loss_bbox']
            #roi_losses.update(extra_bbox_results['loss_extra_bbox'])
            roi_losses['loss_extra_bbox']=extra_bbox_results['loss_extra_bbox']['loss_extra_bbox']
            roi_losses['loss_extra_cls'] = extra_bbox_results['loss_extra_bbox']['loss_extra_cls']
        return roi_losses

    # def _bbox_forward(self, x, rois):
    #     """Box head forward function used in both training and testing."""
    #     # TODO: a more flexible way to decide which feature maps to use
    #     bbox_feats = self.bbox_roi_extractor(
    #         x[:self.bbox_roi_extractor.num_inputs], rois)
    #     if self.with_shared_head:
    #         bbox_feats = self.shared_head(bbox_feats)
    #     cls_score, bbox_pred = self.bbox_head(bbox_feats)
    #
    #     # B: da_cls_score
    #     grl_lambda = 1.0 #B
    #     # # Training progress and GRL lambda
    #     # p = float(batch_idx + epoch_idx * max_batches) / (n_epochs * max_batches)
    #     # grl_lambda = 2. / (1. + np.exp(-10 * p)) - 1
    #     reverse_bbox_feats = GradientReversalFn.apply(bbox_feats, grl_lambda) #B
    #     extra_cls_score, extra_bbox_pred = self.extra_bbox_head(reverse_bbox_feats) #B
    #
    #     bbox_results = dict(
    #         cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats, extra_cls_score= extra_cls_score, extra_bbox_pred = extra_bbox_pred)
    #     return bbox_results

    # def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
    #                         img_metas):
    #     """Run forward function and calculate loss for box head in training."""
    #     rois = bbox2roi([res.bboxes for res in sampling_results])
    #     bbox_results = self._bbox_forward(x, rois)
    #
    #     bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
    #                                               gt_labels, self.train_cfg)
    #     loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
    #                                     bbox_results['bbox_pred'], rois,
    #                                     *bbox_targets)
    #
    #     extra_loss_bbox = self.extra_bbox_head.loss(bbox_results['extra_cls_score'],
    #                                     bbox_results['extra_bbox_pred'], rois,
    #                                     *bbox_targets)#TODO: B replace bbox_targets[0] by other labels
    #
    #     loss_bbox.update(loss_cls_extra=extra_loss_bbox['loss_cls']) #TODO: B
    #     bbox_results.update(loss_bbox=loss_bbox)
    #     bbox_results.update(loss_bbox_extra=extra_loss_bbox)
    #     return bbox_results
    def probe_keep_debug_results_extra_bbox_forward_train(self, extra_bbox_results, bbox_targets):
        if (hasattr(self, 'keep_debug_results')):
            if (self.keep_debug_results):
                if(not hasattr(self,'debug_results')):
                    self.debug_results = {}
                #self.debug_results['extra_features'] = extra_bbox_results['bbox_feats'].clone().cpu().detach().numpy()
                self.debug_results['extra_labels'] = bbox_targets[0].clone().cpu().detach().numpy()

    def _extra_bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                                  img_metas):
        # B: almost identical to StandardRoIHead._bbox_forward_train: only bbox--> extra_bbox
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        extra_bbox_results = self._extra_bbox_forward(x, rois)

        extra_label = None

        if(self.extra_head_annotation_per_image):
            # replace sampling_results[0].pos_gt_labels[:] by extra label
            for i,sr in enumerate(sampling_results):
                #extra_label = img_metas[i][self.extra_label] #TEMP can also get it from gt_labels[i]
                extra_label = gt_labels[i][0] #get it from gt_labels[i]
                sr.pos_gt_labels[:]=extra_label

        else:
            raise NotImplementedError



        bbox_targets = self.extra_bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        if (self.extra_head_annotation_per_image):
            #apply self.extra_head_image_instance_weight to bbox_targets[1] based on bbox_targets[0]
            bckg_class = self.extra_bbox_head.num_classes
            bbox_targets[1][bbox_targets[0]==bckg_class] = self.extra_head_image_instance_weight[0]#weight for background (negative)
            bbox_targets[1][bbox_targets[0] != bckg_class] = self.extra_head_image_instance_weight[1]#weight for positive

            # fix bbox_targets[0], use same label for all targets (both negative and positive)
            bbox_targets[0][bbox_targets[0] == bckg_class] = extra_label

        else:
            raise NotImplementedError


        loss_extra_bbox = self.extra_bbox_head.loss(extra_bbox_results['cls_score'],
                                        extra_bbox_results['bbox_pred'], rois,
                                        *bbox_targets)
        # B: debug vizualization
        # B: TODO: make one line
        self.probe_keep_debug_results_extra_bbox_forward_train(extra_bbox_results, bbox_targets)

        extra_bbox_results.update(loss_extra_bbox=loss_extra_bbox)
        return extra_bbox_results

    def _extra_bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)

        # B: da_cls_score
        # Training progress and GRL lambda
        grl_lambda = self.calc_grl_lambda()
        self.last_grl_lambda = grl_lambda

        if(self.extra_head_with_grad_reversal):
            bbox_feats = GradientReversalFn.apply(bbox_feats, grl_lambda)  # B:plug in GR

        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        extra_cls_score, extra_bbox_pred = self.extra_bbox_head(bbox_feats)  # B

        bbox_results = dict(
            cls_score=extra_cls_score, bbox_pred=extra_bbox_pred, bbox_feats=bbox_feats)
        return bbox_results


