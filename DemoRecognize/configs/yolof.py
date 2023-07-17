dataset_type = 'CocoDataset'
data_root = 'data/coco/'
metainfo = dict(
    classes=('with_mask', 'without_mask', 'mask_weared_incorrect'),
    palette=[(0, 195, 0), (23, 105, 255), (225, 40, 40)])
backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(
        1333,
        800,
    ), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomShift', prob=0.5, max_shift_px=32),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(
        1333,
        800,
    ), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        )),
]
train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='CocoDataset',
        metainfo=dict(
            classes=('with_mask', 'without_mask', 'mask_weared_incorrect'),
            palette=[(0, 195, 0), (23, 105, 255), (225, 40, 40)]),
        ann_file='/content/drive/MyDrive/CS338_DoAn/dataset/annotation/train.json',
        data_prefix=dict(img='/content/drive/MyDrive/CS338_DoAn/dataset/train'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', scale=(
                1333,
                800,
            ), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(type='RandomShift', prob=0.5, max_shift_px=32),
            dict(type='PackDetInputs'),
        ],
        backend_args=None))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        metainfo=dict(
            classes=('with_mask', 'without_mask', 'mask_weared_incorrect'),
            palette=[(0, 195, 0), (23, 105, 255), (225, 40, 40)]),
        ann_file='/content/drive/MyDrive/CS338_DoAn/dataset/annotation/val.json',
        data_prefix=dict(img='/content/drive/MyDrive/CS338_DoAn/dataset/val'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='Resize', scale=(
                1333,
                800,
            ), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PackDetInputs',
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                )),
        ],
        backend_args=None))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        metainfo=dict(
            classes=('with_mask', 'without_mask', 'mask_weared_incorrect'),
            palette=[(0, 195, 0), (23, 105, 255), (225, 40, 40)]),
        ann_file='/content/drive/MyDrive/CS338_DoAn/dataset/annotation/test.json',
        data_prefix=dict(img='/content/drive/MyDrive/CS338_DoAn/dataset/test'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='Resize', scale=(
                1333,
                800,
            ), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PackDetInputs',
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                )),
        ],
        backend_args=None))
val_evaluator = dict(
    type='CocoMetric',
    ann_file='/content/drive/MyDrive/CS338_DoAn/dataset/annotation/val.json',
    metric='bbox',
    format_only=False,
    backend_args=None)
test_evaluator = dict(
    type='CocoMetric',
    ann_file='/content/drive/MyDrive/CS338_DoAn/dataset/annotation/test.json',
    metric='bbox',
    format_only=False,
    backend_args=None)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=32, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.00066667,
        by_epoch=False,
        begin=0,
        end=1500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=32,
        by_epoch=True,
        milestones=[],
        gamma=0.1),
]
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        custom_keys=dict(backbone=dict(lr_mult=0.3333333333333333))))
auto_scale_lr = dict(enable=False, base_batch_size=64)
default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ],
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
model = dict(
    type='YOLOF',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[
            103.53,
            116.28,
            123.675,
        ],
        std=[
            1.0,
            1.0,
            1.0,
        ],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron/resnet50_caffe')),
    neck=dict(
        type='DilatedEncoder',
        in_channels=2048,
        out_channels=512,
        block_mid_channels=128,
        num_residual_blocks=4,
        block_dilations=[
            2,
            4,
            6,
            8,
        ]),
    bbox_head=dict(
        type='YOLOFHead',
        num_classes=3,
        in_channels=512,
        reg_decoded_bbox=True,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[
                1.0,
            ],
            scales=[
                1,
                2,
                4,
                8,
                16,
            ],
            strides=[
                32,
            ]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
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
            add_ctr_clamp=True,
            ctr_clamp=32),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0)),
    train_cfg=dict(
        assigner=dict(
            type='UniformAssigner', pos_ignore_thr=0.15, neg_ignore_thr=0.7),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))
launcher = 'none'
work_dir = '/content/drive/MyDrive/CS338_DoAn/models/YOLOF/epoch'