_base_ = '../yolo/yolov3_d53_8xb8-ms-608-273e_coco.py'
# model settings
model = dict(
    bbox_head=dict(
        num_classes=1))
checkpoint_config = dict(interval=3)
# dataset settings
img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
# Modify dataset related settings
classes = ('drone',)
data_root = '/home/ubuntu/workspace/datasets/final_dataset/'
load_from = 'checkpoints/yolov3_d53_mstrain-608_273e_coco-139f5633.pth'
# lr_config = dict(
#     step=[75, 90])
# # runtime settings
# total_epochs = 100
evaluation = dict(interval=1, metric=['bbox'])

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
total_epochs = 24

auto_scale_lr = dict(base_batch_size=16,enable=True)

dataset_type='CocoDataset'

# dataloader settings
_num_classes = 1
classes = ('drone')
_batch_size = 4
_num_workers = 4

train_dataloader = dict(
    batch_size=_batch_size,
    num_workers=_num_workers,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='annotations/test.json',
        data_prefix=dict(img='test/')
    )
)
val_dataloader = dict(
    batch_size=_batch_size,
    num_workers=_num_workers,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        test_mode=True,
        metainfo=dict(classes=classes),
        ann_file='annotations/test.json',
        data_prefix=dict(img='test/')
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/test.json',
    metric='bbox',
    classwise=True,
    format_only=False,
    backend_args=None)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/test.json',
    metric='bbox',
    format_only=True,
    outfile_prefix='./outputs/yolo3_anti',
    backend_args=None)

model = dict(
    bbox_head=dict(num_classes=_num_classes))

# Runtime settings
# train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
resume = False
checkpoint=dict(type='CheckpointHook', interval=1)
vis_backends = [dict(type='LocalVisBackend'),dict(type='WandbVisBackend'),]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
visualization=dict( # user visualization of validation and test results
    type='DetVisualizationHook',
    draw=True,
    interval=1,
    show=True)