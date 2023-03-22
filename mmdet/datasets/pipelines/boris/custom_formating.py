# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Sequence

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.pipelines.formating import Collect

from ...builder import PIPELINES

@PIPELINES.register_module()
class AddFieldToImgMetas:
    """
    Used to add new fields to results['img_metas'].data
    For example if you add to train pipeline after Collect
    dict(type='AddFieldToImgMetas', fields = ['ignore_negatives'], values = [1], replace = False)

    See also Collect
    """

    def __init__(self,
                 fields = [], # fields to add
                 values=[], # value of each field
                 replace = True): # if such field alrady exists, replace ?
        self.fields = fields
        self.values = values
        self.replace = replace

    def __call__(self, results):
        if('img_metas' not in results):
            raise AssertionError('img_metas not in results')
        for f,v in zip(self.fields, self.values):
            if not(f in results['img_metas'].data and self.replace==False):
                results['img_metas'].data[f] = v


        return results

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(fields={self.fields}, values={self.values}, replace={self.replace})'

@PIPELINES.register_module()
class RobustCollect(Collect):
    '''
    Exactly like Collect but not crashing if key or meta_key not exist
    '''
    def __call__(self, results):
        """
        Exactly like Collect but can skip non-existant fields rather than crash
        """

        data = {}
        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
            # else:
            #     img_meta[key] = None
        data['img_metas'] = DC(img_meta, cpu_only=True)
        for key in self.keys:
            if key in results:
                data[key] = results[key]
            # else:
            #     data[key] = None
        return data


