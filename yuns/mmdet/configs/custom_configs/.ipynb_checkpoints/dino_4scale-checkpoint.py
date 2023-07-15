# The new config inherits a base config to highlight the necessary modification
_base_ = '../dino/dino-4scale_r50_8xb2-36e_coco.py'
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/dino/dino-4scale_r50_8xb2-12e_coco/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth'
auto_scale_lr = dict(base_batch_size=16,enable=True)
# dataset settings
data_root = '/home/ubuntu/workspace/datasets/dacon/coco/'
dataset_type='CocoDataset'

# dataloader settings
_num_classes = 34
classes = ('chevrolet_malibu_sedan_2012_2016'
                ,'chevrolet_malibu_sedan_2017_2019'
                ,'chevrolet_spark_hatchback_2016_2021'
                ,'chevrolet_trailblazer_suv_2021_'
                ,'chevrolet_trax_suv_2017_2019'
                ,'genesis_g80_sedan_2016_2020'
                ,'genesis_g80_sedan_2021_'
                ,'genesis_gv80_suv_2020_'
                ,'hyundai_avante_sedan_2011_2015'
                ,'hyundai_avante_sedan_2020_'
                ,'hyundai_grandeur_sedan_2011_2016'
                ,'hyundai_grandstarex_van_2018_2020'
                ,'hyundai_ioniq_hatchback_2016_2019'
                ,'hyundai_sonata_sedan_2004_2009'
                ,'hyundai_sonata_sedan_2010_2014'
                ,'hyundai_sonata_sedan_2019_2020'
                ,'kia_carnival_van_2015_2020'
                ,'kia_carnival_van_2021_'
                ,'kia_k5_sedan_2010_2015'
                ,'kia_k5_sedan_2020_'
                ,'kia_k7_sedan_2016_2020'
                ,'kia_mohave_suv_2020_'
                ,'kia_morning_hatchback_2004_2010'
                ,'kia_morning_hatchback_2011_2016'
                ,'kia_ray_hatchback_2012_2017'
                ,'kia_sorrento_suv_2015_2019'
                ,'kia_sorrento_suv_2020_'
                ,'kia_soul_suv_2014_2018'
                ,'kia_sportage_suv_2016_2020'
                ,'kia_stonic_suv_2017_2019'
                ,'renault_sm3_sedan_2015_2018'
                ,'renault_xm3_suv_2020_'
                ,'ssangyong_korando_suv_2019_2020'
                ,'ssangyong_tivoli_suv_2016_2020'
            )
_batch_size = 4
_num_workers = 4

train_dataloader = dict(
    batch_size=_batch_size,
    num_workers=_num_workers,
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='annotations/train.json',
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
        data_root=data_root,
        ann_file='annotations/val.json',
        data_prefix=dict(img='val/')
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
    ann_file=data_root + 'annotations/val.json',
    metric='bbox',
    classwise=True,
    format_only=False,
    backend_args=None)
test_evaluator = val_evaluator
# test_evaluator = dict(
#     type='CocoMetric',
#     ann_file=data_root + 'annotations/test.json',
#     metric='bbox',
#     format_only=True,
#     outfile_prefix='./outputs/dino_4scale',
#     backend_args=None)

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