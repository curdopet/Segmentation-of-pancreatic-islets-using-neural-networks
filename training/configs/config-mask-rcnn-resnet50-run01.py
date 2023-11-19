_base_ = [
    '../mmdetection/configs/_base_/models/mask-rcnn_r50_fpn.py',
    '../mmdetection/configs/_base_/datasets/coco_instance.py',
    '../mmdetection/configs/_base_/schedules/schedule_1x.py',
    '../mmdetection/configs/_base_/default_runtime.py',
]

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1), mask_head=dict(num_classes=1)))

# Modify dataset related settings
data_root = 'data_split/'
metainfo = {
    'classes': ('islet',),
}
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='jsons/coco-format-training-islets-only-ge10.json',
        data_prefix=dict(img='training_data/inputs/')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='jsons/coco-format-validation-islets-only-ge10.json',
        data_prefix=dict(img='validation_data/inputs/')))
test_dataloader = val_dataloader

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + 'jsons/coco-format-validation-islets-only-ge10.json')
test_evaluator = val_evaluator

train_cfg=dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)

