#test ExtraFormatBundle, AddFieldToImgMetas, RobustCollect

from mmdet.datasets.pipelines.boris.custom_formating import AddFieldToImgMetas, RobustCollect
from mmcv.parallel import DataContainer as DC

def test_AddFieldToImgMetas():
    my_add_fileds = AddFieldToImgMetas(fields=['f1', 'f2'], values=[2, 'text'], replace=True)
    results = {'k1':1, 'k2':2, 'k3':3, 'k4':4, 'k5':5}
    img_meta = {}
    for key in ['k1','k5','k6']:
        if key in results:
            img_meta[key] = results[key]

    results['img_metas'] = DC(img_meta, cpu_only=True)


    results = my_add_fileds.__call__(results)

    #assert existance of ['f1', 'f2']
    for k in ['f1', 'f2']:
        assert (k in results['img_metas'].data)

def test_RobustCollect():
    my_collect = RobustCollect(keys=['k1','k2', 'k3_not_exist'], meta_keys=['f1','f2','f_not_exist'])
    results = {'k1': 1, 'k2': 2, 'k3': 3, 'k4': 4, 'k5': 5, 'f1': [1,2], 'f2': 23, 'f3': 'gone'}
    results = my_collect.__call__(results)
    # assert existance of ['f1', 'f2'] etc
    for k in ['f1', 'f2']:
        assert (k in results['img_metas'].data)
    for k in ['f_not_exist', 'f3']:
        assert (k not in results['img_metas'].data)
    for k in ['k1', 'k2']:
        assert (k in results)
    for k in ['k3', 'k3_not_exist']:
        assert (k not in results)


if __name__ == "__main__":
    test_AddFieldToImgMetas()
    test_RobustCollect()