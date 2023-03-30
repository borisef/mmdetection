
extra_label_in_json = 'domain_id' # should be present in image (and target)
meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'img_norm_cfg',
                            'extra_label_name', # new
                            extra_label_in_json) #domain_id is new, should be like in json

num_classes = 12
CLASSES = ['obstacles', 'biker', 'car', 'pedestrian', 'trafficLight', 'trafficLight-Green', 'trafficLight-GreenLeft',
           'trafficLight-Red','trafficLight-RedLeft','trafficLight-Yellow','trafficLight-YellowLeft','truck']

num_domains = 3
DOMAINS = ['grayblur', 'negative',  'rgb'] # for vizualisation

work_dir = '/home/borisef/projects/mmdetHack/Runs/try_simplify_config_01'
model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4),  # like num strides
    rpn_head=dict(
        type='RPNHeadWithWeightPerImage',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8, 16],
            ratios=[0.5,  1.0, 2.0],
            strides=[4, 8, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        #type='StandardRoIHead',
        type='StandardRoIHeadWithExtraBBoxHead',#Like StandardRoIHead + extra bbox head

        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHeadWithWeightPerImage',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=num_classes,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        ),
        # extra head params (one dictionary)
        extra_head_params=dict(
            with_grad_reversal=False,
            image_instance_weight=[0.2, 1],  # for domain adaptation weighting
            lambda_params=dict(start_end_max_epoch=[0, 8, 8], iters_per_epoch=50, power_factor=3.0, default_lambda=None,
                               init_epoch=2)
        ),
        extra_bbox_head=dict(
                    type='Shared2FCBBoxHead',
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=num_domains,
                    bbox_coder=dict(
                        type='DeltaXYWHBBoxCoder',
                        target_means=[0.0, 0.0, 0.0, 0.0],
                        target_stds=[0.1, 0.1, 0.2, 0.2]),
                    reg_class_agnostic=False,
                    loss_cls=dict(
                        type='CrossEntropyLoss',
                        use_sigmoid=False,
                        class_weight=None,
                        ignore_index=None,
                        loss_weight=1.0), # D.A.
                    loss_bbox=dict(type='L1Loss', loss_weight=0.0),
                ),
        ),

    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.001,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)))

#data_root = 'car_damage/'
img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='LoadExtraAnnotations', skip_on_error = False),  # load any extra annotations
    dict(
        type='Resize',
        img_scale=[(1333, 400), (1333, 450), (1333, 550), (1333, 500),
                   (1333, 500), (1333, 550)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='ExtraFormatBundle', key_names=['gt_extra_labels']),
    dict(type='Collect', meta_keys=meta_keys, keys=['img', 'gt_bboxes', 'gt_labels', 'gt_extra_labels'])
]

test_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[103.53, 116.28, 123.675],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False),
                    dict(type='Pad', size_divisor=32),
                    #dict(type='ImageToTensor', keys=['img']),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]
train_pipeline1 = train_pipeline + [
    dict(type='AddFieldToImgMetas', fields = ['rpn_head.loss_weight'], values = [100.01], replace = True),
    dict(type='AddFieldToImgMetas', fields=['roi_head.loss_weight'], values=[0.0], replace=True),
]

train_pipeline2 = train_pipeline + [
    dict(type='AddFieldToImgMetas', fields = ['ignore_negatives'], values = [1], replace = False),
    dict(type='AddFieldToImgMetas', fields = ['rpn_head.loss_weight'], values = [0.0], replace = True),
    dict(type='AddFieldToImgMetas', fields=['roi_head.loss_weight'], values=[100.01], replace=True),
]
train_data1 = dict(
    type='MyCocoDataset',
    ann_file='set_500_mixed_domains3/data_da.json',
    img_prefix='set_500_mixed_domains3/',
    classes=CLASSES,
    pipeline=train_pipeline1,
    data_root='/home/borisef/datasets/traffic/',
    extra_label=extra_label_in_json)

train_data2 = dict(
    type='MyCocoDataset',
    ann_file='set_500_mixed_domains3/data_da.json',
    img_prefix='set_500_mixed_domains3/',
    classes=CLASSES,
    pipeline=train_pipeline2,
    data_root='/home/borisef/datasets/traffic/',
    extra_label=extra_label_in_json)

train_data = dict(
    type = 'ConcatDataset',
    datasets = [train_data1, train_data2],
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
   train = train_data,
    val=dict(
        type='CocoDataset',
        ann_file='set_50_domain_grayblur/data_da.json',
        img_prefix='set_50_domain_grayblur/',
        classes = CLASSES,
        pipeline=test_pipeline,
        data_root='/home/borisef/datasets/traffic/'),
    test=dict(
        type='CocoDataset',
        ann_file='set_50_domain_grayblur/data_da.json',
        img_prefix='set_50_domain_grayblur/',
        classes=CLASSES,
        pipeline=test_pipeline,
        data_root='/home/borisef/datasets/traffic/'))

#evaluation = dict(interval=12, metric='mAP')
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup=None,
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[1,100])
runner = dict(type='EpochBasedRunner', max_epochs=500)
checkpoint_config = dict(interval=10)
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from =  '../checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
resume_from = work_dir + '/latest.pth'
#workflow = [('train', 1), ('val', 1)] #not working
workflow = [('train', 2)]

seed = 0
gpu_ids = range(0, 1)

viz_debugHookRoIHead = dict(
    type='VizDebugFeaturesHookStandardRoIHead',
    num_classes=num_classes,
    class_names=CLASSES,
    log_folder=work_dir + '/class_features_roihead',
    max_per_class=150,
    #epochs = [0,10, 25,50,75,200,295] #
)

viz_debugHookRoIHeadExtra = dict(
    type='VizDebugFeaturesHookStandardRoIHeadWithExtraBBoxHead',
    num_classes=[num_classes, num_domains],
    class_names=[CLASSES, DOMAINS],
    log_folder=work_dir + '/class_features_roihead_with_extra',
    max_per_class=150,
    log_lambda_to_runner = True,
    epochs = [0,25,50,75,200,295] #
)



fmHook = dict(
    conv_filters = ['backbone.conv1'],
    relu_filters = ['backbone.relu'],
    outDir = work_dir + '/exp_out_fm',
    imName = None,
    type = 'FeatureMapHook'
)

custom_hooks = [ viz_debugHookRoIHead, viz_debugHookRoIHeadExtra]#epoch_to_model_hook,


custom_imports=dict(
    imports=['boris.kitti_Dataset',
             'boris.viz_debug_features_hook',
             'boris.standard_roi_head_with_extra',
             'boris.experimental_hook',
             'boris.get_feature_maps_hook',
             #'boris.set_epoch_data_in_model_hook',
             'mmdet.datasets.pipelines.boris.custom_formating',
             'mmdet.models.dense_heads.boris.custom_rpn_head',
             'mmdet.models.roi_heads.bbox_heads.boris.custom_bbox_head',
             'boris.user_loading',
             'boris.user_loading',
             'boris.my_coco',
             'boris.user_formating'])

example_images = '/home/borisef/datasets/traffic/set_50_domain_grayblur/'

