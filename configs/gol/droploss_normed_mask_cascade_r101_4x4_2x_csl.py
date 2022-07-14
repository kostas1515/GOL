_base_ = [
    '../_base_/models/cascade_mask_rcnn_r50_fpn.py',
    '../_base_/datasets/lvis_v1_instance.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')),
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1203,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='DropLoss',
                    use_sigmoid=True,
                    loss_weight=1.0,
                    lambda_=0.0011,
                    version='v1',
                    use_classif='cls_rel'),
                init_cfg=dict(
                    type='Constant',
                    val=0.001,
                    bias=-2.75,
                    override=dict(name='fc_cls')),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1203,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='DropLoss',
                    use_sigmoid=True,
                    loss_weight=1.0,
                    lambda_=0.0011,
                    version='v1',
                    use_classif='cls_rel'),
                init_cfg=dict(
                    type='Constant',
                    val=0.001,
                    bias=-2.75,
                    override=dict(name='fc_cls')),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1203,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='DropLoss',
                    use_sigmoid=True,
                    loss_weight=1.0,
                    lambda_=0.0011,
                    version='v1',
                    use_classif='cls_rel'),
                init_cfg=dict(
                    type='Constant',
                    val=0.001,
                    bias=-2.75,
                    override=dict(name='fc_cls')),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ],
        mask_head=dict(num_classes=1203,predictor_cfg=dict(type='NormedConv2d', tempearture=20))),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.0001,
            nms=dict(type='nms', iou_threshold=0.5),
            # LVIS allows up to 300
            max_per_img=300)))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
data = dict(train=dict(
    oversample_thr=0.0,
    dataset=dict(pipeline=train_pipeline)))
evaluation = dict(interval=24, metric=['bbox', 'segm'])

work_dir = 'experiments/droploss_normed_mask_cascade_r101_4x4_2x_csl'
# work_dir = 'experiments/test'