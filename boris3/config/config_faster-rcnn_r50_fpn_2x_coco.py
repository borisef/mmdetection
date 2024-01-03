#CLASSES = ('vehicles', 'Ambulance', 'Bus', 'car', 'Motorcycle','Truck')
CLASSES = ['Ambulance', 'Bus', 'Car', 'Motorcycle','Truck']

MY_NUM_CLASSES = 5
log_level = 'INFO'
load_from = '/home/borisef/data/mmdet_models/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'
resume = False
launcher = 'none'
work_dir = './work_dirs/faster-rcnn_r50_fpn_2x_vehicles'
my_batch_size = 2

if(resume):
    load_from = None

dataset_type = 'CocoDataset'
my_data_root = '/home/borisef/data/vehicles/'
data_root = my_data_root


auto_scale_lr = dict(base_batch_size=16, enable=True)
backend_args = None
default_scope = 'mmdet'

train_cfg = dict(max_epochs=24, type='EpochBasedTrainLoop', val_interval=2)
test_cfg = dict(type='TestLoop')
val_cfg = dict(type='ValLoop')

vis_backends = [
    dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')
]

default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook',
                       interval = 5,
                       draw = True,
                       test_out_dir = work_dir + '/test_out_dir'))

env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))

log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)

model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNet'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        type='FPN'),
    roi_head=dict(
        bbox_head=dict(
            bbox_coder=dict(
                target_means=[
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                target_stds=[
                    0.1,
                    0.1,
                    0.2,
                    0.2,
                ],
                type='DeltaXYWHBBoxCoder'),
            fc_out_channels=1024,
            in_channels=256,
            loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
            loss_cls=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            num_classes=MY_NUM_CLASSES,
            reg_class_agnostic=False,
            roi_feat_size=7,
            type='Shared2FCBBoxHead'),
        bbox_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        type='StandardRoIHead'),
    rpn_head=dict(
        anchor_generator=dict(
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            scales=[
                8,
            ],
            strides=[
                4,
                8,
                16,
                32,
                64,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
        loss_cls=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        type='RPNHead'),
    test_cfg=dict(
        rcnn=dict(
            max_per_img=100,
            nms=dict(iou_threshold=0.5, type='nms'),
            score_thr=0.05),
        rpn=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=1000)),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=False,
                min_pos_iou=0.5,
                neg_iou_thr=0.5,
                pos_iou_thr=0.5,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=-1,
                num=512,
                pos_fraction=0.25,
                type='RandomSampler')),
        rpn=dict(
            allowed_border=-1,
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.3,
                neg_iou_thr=0.3,
                pos_iou_thr=0.7,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=False,
                neg_pos_ub=-1,
                num=256,
                pos_fraction=0.5,
                type='RandomSampler')),
        rpn_proposal=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=2000)),
    type='FasterRCNN')
optim_wrapper = dict(
    optimizer=dict(lr=0.02, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=24,
        gamma=0.1,
        milestones=[
            16,
            22,
        ],
        type='MultiStepLR'),
]
test_pipeline = [
    dict(backend_args=backend_args, type='LoadImageFromFile'),
    dict(keep_ratio=True,
         scale=(600,600),
         type='Resize'),
    dict(type = 'RandomCrop',
         crop_size=(384, 384),
         allow_negative_crop=False),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
val_pipeline = test_pipeline
train_pipeline = [
    dict(backend_args=backend_args, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=True,
        #  scale=(
        # 416,
        # 416,),
         scale_factor = 1.5, type='Resize'),
    dict(type = 'RandomCrop',
         crop_size=(384, 600),
         allow_negative_crop=False),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
test_dataloader = dict(
    batch_size=my_batch_size,
    dataset=dict(
        ann_file='test/annotations_5classes.json',
        backend_args=backend_args,
        metainfo=dict(classes=CLASSES),
        data_prefix=dict(img='test/'),
        data_root=my_data_root,
        pipeline=test_pipeline,
        test_mode=True,
        type=dataset_type),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))

test_evaluator = dict(
    ann_file= my_data_root + 'test/annotations_5classes.json',
    backend_args=backend_args,
    format_only=False,
    metric='bbox',
    type='CocoMetric')

train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=my_batch_size,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=CLASSES),
        ann_file='test/annotations_5classes.json',
        data_prefix=dict(img='test/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))


val_dataloader = dict(
    batch_size=my_batch_size,
    dataset=dict(
        ann_file=my_data_root+'test/annotations_5classes.json',
        backend_args=backend_args,
        metainfo=dict(classes=CLASSES),
        data_prefix=dict(img='test/'),
        data_root=my_data_root,
        pipeline=val_pipeline,
        test_mode=True,
        type=dataset_type),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))

val_evaluator = test_evaluator

vis_backends = [
    dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    save_dir = work_dir + '/vizu'
)

example_images = my_data_root+'/test' #for output dir