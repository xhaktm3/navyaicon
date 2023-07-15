_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'
]

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=1,)))
checkpoint_config = dict(interval=3)

classes = ('drone',)

load_from = '/home/ubuntu/workspace/yuns/models/'
optimizer = dict(lr=0.001)

_batch_size = 3
_num_workers = 4

train_dataloader = dict(
    batch_size=_batch_size,
    num_workers=_num_workers,
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        ann_file='/home/ubuntu/workspace/datasets/annotations/eo_test1.json',
        data_prefix=dict(img='train/')
    )
)
val_dataloader = dict(
    batch_size=_batch_size,
    num_workers=_num_workers,
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        metainfo=dict(classes=classes),
        ann_file='/home/ubuntu/workspace/datasets/annotations/eo_test1.json',
        data_prefix=dict(img='train/')
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file='/home/ubuntu/workspace/datasets/annotations/eo_test1.json',
    metric='bbox',
    classwise=True,
    format_only=False,
    backend_args=None)
test_evaluator = dict(
    type='CocoMetric',
    ann_file='/home/ubuntu/workspace/datasets/annotations/eo_test1.json',
    metric='bbox',
    format_only=True,
    outfile_prefix='./outputs/dino_eo_re',
    backend_args=None)

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