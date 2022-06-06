import torch
from mmdet.models.builder import HEADS, build_head, build_roi_extractor
from mmdet.core import bbox2roi
from  mmdet.models.roi_heads.standard_roi_head import StandardRoIHead


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

