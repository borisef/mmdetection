from mmdet.datasets.pipelines.formating import PIPELINES, to_tensor, DC

@PIPELINES.register_module()
class ExtraFormatBundle:
    """
    Like FormatBundle but names are given in key_names
    """

    def __init__(self, key_names= ['gt_extra_labels']):
        self.key_names = key_names

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with \
                default bundle.
        """
        for key in self.key_names:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]))

        return results

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(key_names={self.key_names})'
