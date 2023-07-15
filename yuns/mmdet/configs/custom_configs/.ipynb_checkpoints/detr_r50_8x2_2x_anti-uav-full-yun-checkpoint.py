# The new config inherits a base config to highlight the necessary modification
_base_ = '../detr/detr_r50_8xb2-150e_coco.py'
load_from = '/home/ubuntu/workspace/yuns/models/detr_r50_8x2_2x_anti-uav-full_model.pth'
auto_scale_lr = dict(base_batch_size=16,enable=True)
# dataset settings
# data_root = '/home/ubuntu/workspace/datasets/dacon/coco/'
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
# test_dataloader = dict(
#     batch_size=_batch_size,
#     num_workers=_num_workers,
#     dataset=dict(
#         type=dataset_type,
#         test_mode=True,
#         metainfo=dict(classes=classes),
#         data_root=data_root,
#         ann_file='annotations/test.json',
#         data_prefix=dict(img='test/')
#     )
# )

# Evaluator settings
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

# Training settings
# train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)

# Model settings
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

# Using TTA
# tta_model = dict(
#     type='DetTTAModel',
#     tta_cfg=dict(nms=dict(
#                    type='nms',
#                    iou_threshold=0.5),
#                    max_per_img=100))

# tta_pipeline = [
#     dict(type='LoadImageFromFile',
#         backend_args=None),
#     dict(
#         type='TestTimeAug',
#         transforms=[[
#             dict(type='Resize', scale=(1333, 800), keep_ratio=True)
#         ], [ # It uses 2 flipping transformations (flipping and not flipping).
#             dict(type='RandomFlip', prob=1.),
#             dict(type='RandomFlip', prob=0.)
#         ], [
#             dict(
#                type='PackDetInputs',
#                meta_keys=('img_id', 'img_path', 'ori_shape',
#                        'img_shape', 'scale_factor', 'flip',
#                        'flip_direction'))
#        ]])]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)