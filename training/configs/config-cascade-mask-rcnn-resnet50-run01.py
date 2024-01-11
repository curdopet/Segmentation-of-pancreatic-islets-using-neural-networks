_base_ = [
    '../../../mmdetection/configs/_base_/models/cascade-mask-rcnn_r50_fpn.py',
    '../../../mmdetection/configs/_base_/datasets/coco_instance.py',
    '../../../mmdetection/configs/_base_/schedules/schedule_1x.py',
    '../../../mmdetection/configs/_base_/default_runtime.py',
]

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    rpn_head=dict(
            anchor_generator=dict(
                scales=[1, 2, 4, 8])),
    oi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ],
        mask_head=dict(num_classes=1)))

# Modify dataset related settings
data_root = '../data_split/'
metainfo = {
    'classes': ('islet',),
}

# Train & Val pipeline
backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(2048, 1536), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(2048, 1536), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# Train & Val dataloader
train_dataloader = dict(
    batch_size=2,
    num_workers=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        pipeline=train_pipeline,
        ann_file='jsons/coco-format-training-islets-only.json',
        data_prefix=dict(img='training_data/inputs/')))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        pipeline=test_pipeline,
        ann_file='jsons/coco-format-validation-islets-only.json',
        data_prefix=dict(img='validation_data/inputs/')))
test_dataloader = val_dataloader

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + 'jsons/coco-format-validation-islets-only.json')
test_evaluator = val_evaluator

# Visualize progress using WandB
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs={
            'project': 'instance-seg-islets',
            'tags': ['cascade-mask-rcnn', 'resnet50'],
            'name': 'cascade-mask-rcnn-resnet50-run01',
         })
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')

# Train config
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
