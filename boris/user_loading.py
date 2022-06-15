from mmdet.datasets.pipelines.loading import PIPELINES
import numpy as np
#BE
@PIPELINES.register_module()
class LoadExtraAnnotations:
    """Load domain annotations.
    Like LoadExtraAnnotations
    but instead of labels load index of domain


    """

    def __init__(self,
                 map_name_to_id = None,
                 annotation_per_image = True,
                 name=None):
        self.name = name
        self.map_name_to_id = map_name_to_id
        self.annotation_per_image = annotation_per_image


    def _load_extra_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """

        # if we suppose domain is one per image
        if(self.annotation_per_image):
            extra_label = results['img_info'][self.name]
            # may be map from name to id
            if(self.map_name_to_id is not None):
                extra_label = self.map_name_to_id[extra_label]

            num_annotations = len(results['ann_info'])
            results['gt_extra_labels'] = np.array([extra_label]*num_annotations,
                                                  dtype=results['gt_labels'].dtype)
            results[self.name] = extra_label#TODO: remove this or gt_extra_label_per_image
            results['gt_extra_label_per_image'] = extra_label
        else: # TODO: implement the case with domain is one per bbox
            #see _parse_ann_info function in coco
            num_annotations = len(results['ann_info'])
            results['gt_extra_labels'] = np.array(results['ann_info']['extra_labels'], dtype=results['gt_labels'].dtype)
            results['gt_extra_label_per_image'] = results['img_info'][self.name]
            results[self.name] = results['img_info'][self.name]
            #raise NotImplementedError

        return results


    def __call__(self, results):
        """
        """

        results = self._load_extra_labels(results)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(annotation_per_image={self.annotation_per_image}, '
        repr_str += f'name={self.name}, '

        return repr_str