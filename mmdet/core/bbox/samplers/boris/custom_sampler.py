# Copyright (c) OpenMMLab. All rights reserved.
import torch

from  mmdet.core.bbox.samplers.base_sampler import BaseSampler
from  mmdet.core.bbox.samplers.random_sampler import  RandomSampler
from mmdet.core.bbox.samplers.ohem_sampler import OHEMSampler
from mmdet.core.bbox.builder import BBOX_SAMPLERS


class BasicSamplerWithIgnore():
    """
    May be used as father class for any sampler (at 1st place)
    To ignore all negative samples if in img_metas there is a field
    img_metas['ignore_negatives'] = True 0
    """
    def _sample_neg(self, assign_result, num_expected, **kwargs):
        call_num_expected = num_expected
        if 'img_metas' in kwargs:
            if('ignore_negatives' in kwargs['img_metas']):
                if(kwargs['img_metas']['ignore_negatives'] == 1):
                    call_num_expected = 0
        """Randomly sample some negative samples."""
        return super()._sample_neg( assign_result, call_num_expected, **kwargs)

@BBOX_SAMPLERS.register_module()
class RandomSamplerWithIgnore(BasicSamplerWithIgnore, RandomSampler):
    """Random sampler with ignore negatives: see BasicSamplerWithIgnore

    """
    pass

@BBOX_SAMPLERS.register_module()
class OHEMSamplerWithIgnore(BasicSamplerWithIgnore, OHEMSampler):
    """Random sampler with ignore negatives: see BasicSamplerWithIgnore

    """
    pass