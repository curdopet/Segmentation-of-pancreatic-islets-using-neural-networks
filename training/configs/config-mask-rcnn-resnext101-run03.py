_base_ = [
    '../../../mmdetection/configs/_base_/models/mask-rcnn_r50_fpn.py',
    '../../../mmdetection/configs/_base_/datasets/coco_instance.py',
    '../../../mmdetection/configs/_base_/schedules/schedule_2x.py',
    '../../../mmdetection/configs/_base_/default_runtime.py',
]

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_32x4d')),
    rpn_head=dict(
            anchor_generator=dict(
                scales=[1, 2, 4, 8])),
    roi_head=dict(
        bbox_head=dict(num_classes=1), mask_head=dict(num_classes=1)))

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
test_evaluator = dict(ann_file=data_root + 'jsons/coco-format-test-islets-only.json')

# Visualize progress using WandB
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs={
            'project': 'instance-seg-islets',
            'tags': ['mask-rcnn', 'resnext101'],
            'name': 'mask-rcnn-resnext101-run03',
         })
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')

# Train config
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)
