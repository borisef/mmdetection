from mmdet.datasets.pipelines.loading import PIPELINES
import numpy as np
#BE
@PIPELINES.register_module()
class LoadExtraAnnotations:
    """Load domain annotations.
    Like LoadAnnotations
    but instead of labels load index of domain or someth like that based on name


    """

    def __init__(self,
                 skip_on_error = False,
                 map_name_to_id = None
                 ):

        self.map_name_to_id = map_name_to_id
        self.skip_on_error = skip_on_error


    def _load_extra_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """
        #extra_name = self.name
        if('extra_label_name' in results['ann_info']):
            extra_name = results['ann_info']['extra_label_name']
        else:
            if(self.skip_on_error):
                return results
            else:
                raise ValueError("No  extra_label found in data within ann_info.\n")

        if(extra_name in results['img_info']):
            extra_label = results['img_info'][extra_name]
            results['extra_label_name'] = extra_name
        else:
            if (self.skip_on_error):
                return results
            else:
                raise AssertionError(extra_name + " is not in image attributes")
        if (self.map_name_to_id is not None):
            extra_label = self.map_name_to_id[extra_label]

        #init all targets
        num_annotations = len(results['ann_info']['labels'])
        results['gt_extra_labels'] = np.array([extra_label] * num_annotations,
                                              dtype=results['gt_labels'].dtype)
        results[extra_name] = extra_label  # TODO: remove this or gt_extra_label_per_image
        results['gt_extra_label_per_image'] = extra_label

        #now replace extra_label per target if exists
        if('extra_labels' in results['ann_info']):
            for i in range(num_annotations):
                if (self.map_name_to_id is not None):
                    results['gt_extra_labels'][i] = self.map_name_to_id[results['ann_info']['extra_labels'][i]]
                else:
                    results['gt_extra_labels'][i] = results['ann_info']['extra_labels'][i]


        return results


    def __call__(self, results):
        """
        """

        results = self._load_extra_labels(results)

        return results

    def __repr__(self):
        # repr_str = self.__class__.__name__
        # repr_str += f'(annotation_per_image={self.annotation_per_image}, '
        # repr_str += f'name={self.name}, '

        return self.__class__.__name__
