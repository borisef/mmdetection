from mmdet.datasets.pipelines.loading import PIPELINES

#BE
@PIPELINES.register_module()
class LoadExtraAnnotations:
    """Load domain annotations.
    Like LoadExtraAnnotations
    but instead of labels load index of domain


    """

    def __init__(self,
                 per_bbox=False,
                 map_name_to_id = None,
                 domain_per_image = True,
                 name=None):
        self.per_bbox = per_bbox
        self.name = name
        self.map_name_to_id = map_name_to_id
        self.domain_per_image = domain_per_image


    def _load_domain_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """

        # we suppose domain is one per image
        if(self.domain_per_image):
            domain = results['img_info'][self.name]
            # may be map from name to id
            if(self.map_name_to_id is not None):
                domain = self.map_name_to_id[domain]

            num_annotations = len(results['ann_info'])
            results['gt_domains'] = [domain]*num_annotations
        else: # TODO: implement the case with doamin is one per bbox
            raise NotImplementedError

        return results


    def __call__(self, results):
        """
        """

        results = self._load_domain_labels(results)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(per_bbox={self.per_bbox}, '
        repr_str += f'name={self.name}, '

        return repr_str