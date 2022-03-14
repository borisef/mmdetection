# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weight_reduce_loss

import mmdet.models.losses.cross_entropy_loss as c_e_l


def _smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing=0.0, transform_smoothing = None):
    assert 0 <= smoothing < 1
    with torch.no_grad():
        targets = torch.empty(size=(targets.size(0), n_classes),
                              device=targets.device) \
            .fill_(smoothing / (n_classes - 1)) \
            .scatter_(1, targets.data.unsqueeze(1), 1. - smoothing)
        if transform_smoothing is not None:
            # targets=targets*transform_smoothing
            targets = targets.matmul(transform_smoothing)



    return targets

#B
def smoothed_loss(pred,
                  label,
                  weight=None,
                  reduction='mean',
                  avg_factor=None,
                  class_weight=None,
                  ignore_index=-100,
                  smoothing=0,
                  smoothing_transform=None,
                  loss_func_to_call=None
                           ):
    """Calculate the CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int | None): The label index to be ignored.
            If None, it will be set to default value. Default: -100.

    Returns:
        torch.Tensor: The calculated loss
    """
    # The default value of ignore_index is the same as F.cross_entropy
    ignore_index = -100 if ignore_index is None else ignore_index
    # element-wise losses
    # loss = F.cross_entropy(
    #     pred,
    #     label,
    #     weight=class_weight,
    #     reduction='none',
    #     ignore_index=ignore_index)

    if(smoothing_transform is not None):
        num_classes = smoothing_transform.shape[0]
    else:
        num_classes = pred.shape[1]
    label_smoo = _smooth_one_hot(label, num_classes,smoothing,smoothing_transform)
    if(loss_func_to_call is None):
        loss_func_to_call = nn.CrossEntropyLoss(weight=class_weight, ignore_index=ignore_index, reduce=None, reduction=reduction, label_smoothing=0)

    loss = loss_func_to_call(pred,label_smoo)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss

@LOSSES.register_module()
class SmoothedLoss(nn.Module):

    def __init__(self,
                 smoothing = 0.1,
                 smoothing_transform = None,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 ignore_index=None,
                 loss_func = 'CrossEntropyLoss',
                 loss_weight=1.0):
        """CrossEntropyLoss.

        Args:
            use_sigmoid (bool, optional): Whether the prediction uses sigmoid
                of softmax. Defaults to False.
            use_mask (bool, optional): Whether to use mask cross entropy loss.
                Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            ignore_index (int | None): The label index to be ignored.
                Defaults to None.
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super(SmoothedLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        self.smoothing = smoothing
        self.smoothing_transform = smoothing_transform
        self.loss_func = loss_func

        if(smoothing_transform is not None):
            self.smoothing_transform = torch.tensor(smoothing_transform, dtype = torch.float32).cuda()

        if self.use_sigmoid:
            assert not self.use_sigmoid, "Not implemented"
        elif self.use_mask:
            assert not self.use_mask, "Not implemented"
        else:
            self.cls_criterion = smoothed_loss

        self.loss_func_to_call = None
        self.ignore_index = -100 if self.ignore_index is None else self.ignore_index

        if self.loss_func is "kl_div":
            def kl(pred,label):
                return nn.functional.kl_div( nn.functional.softmax(pred, dim=1), label,  reduction = 'batchmean')


            self.loss_func_to_call = kl #TODO nn.functional.kl_div(nn.functional.softmax(pred, dim=2),label1, reduction = 'batchmean')
            assert False, "Not implemented kl_div"
        elif self.loss_func is "MultiLabelSoftMarginLoss":
            self.loss_func_to_call = nn.MultiLabelSoftMarginLoss(weight=self.class_weight, reduction =self.reduction)
        elif self.loss_func is "CrossEntropyLoss":
            self.loss_func_to_call = nn.CrossEntropyLoss(weight=self.class_weight, ignore_index = self.ignore_index, reduction =self.reduction)
        elif self.loss_func is "BCEWithLogitsLoss":
            self.loss_func_to_call = nn.MultiLabelSoftMarginLoss(weight=self.class_weight, reduction =self.reduction)


    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=None,
                **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The prediction.
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The method used to reduce the
                loss. Options are "none", "mean" and "sum".
            ignore_index (int | None): The label index to be ignored.
                If not None, it will override the default value. Default: None.
        Returns:
            torch.Tensor: The calculated loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if ignore_index is None:
            ignore_index = self.ignore_index

        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(
                self.class_weight, device=cls_score.device)
        else:
            class_weight = None
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_index=ignore_index,
            smoothing = self.smoothing,
            smoothing_transform = self.smoothing_transform,
            loss_func_to_call = self.loss_func_to_call,
            **kwargs)
        return loss_cls

