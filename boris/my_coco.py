from mmdet.datasets import CocoDataset
from mmdet.datasets.builder import DATASETS
import numpy as np

@DATASETS.register_module()
class MyCocoDataset(CocoDataset):
    #init
    def __init__(self, extra_label  = None,  **kwargs):
       super(MyCocoDataset, self).__init__(**kwargs)
       self.extra_label = extra_label

    def _parse_ann_info(self, img_info, ann_info):
        #call  CocoDataset
        ann1 = super(MyCocoDataset, self)._parse_ann_info(img_info, ann_info)
        ann2 = self._parse_ann_info_only_extra_info(img_info, ann_info)
        ann1.update(ann2)
        return ann1

    def _parse_ann_info_only_extra_info(self, img_info, ann_info):
        """Parse extra_labels ONLY.
        Remark: some code duplication from _parse_ann_info_ because we have extra_label only for
        targets with label/bbox

        """
        if(self.extra_label is None):
            return dict()

        gt_bboxes = []
        gt_extra_labels = []
        gt_bboxes_ignore = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue

            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                if(self.extra_label in ann):
                    gt_extra_labels.append(ann[self.extra_label])

        if gt_bboxes:
            gt_extra_labels = np.array(gt_extra_labels, dtype=np.int64)
        else:
            gt_extra_labels = np.array([], dtype=np.int64)

        ann = dict(
            extra_labels= gt_extra_labels
        )
        if(len(gt_extra_labels) == 0):
            ann = dict()

        return ann
