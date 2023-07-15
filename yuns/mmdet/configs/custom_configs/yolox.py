# The new config inherits a base config to highlight the necessary modification
_base_ = '../yolox/yolox_x_8xb8-300e_coco.py'

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
_batch_size = 2
_num_workers = 1

train_dataset =dict(
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='annotations/train_sample_10.json',
        data_prefix=dict(img='train/'),
    )
)
train_dataloader = dict(
    batch_size=_batch_size,
    num_workers=_num_workers,
    dataset=train_dataset
)

val_dataloader = dict(
    batch_size=_batch_size,
    num_workers=_num_workers,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='annotations/train_sample_10.json',
        data_prefix=dict(img='train/'),
        test_mode=True))

# val_dataloader = dict(
#     batch_size=_batch_size,
#     num_workers=_num_workers,
#     dataset=dict(
#         type=dataset_type,
#         test_mode=True,
#         metainfo=dict(classes=classes),
#         data_root=data_root,
#         ann_file='annotations/train_sample_10.json',
#         data_prefix=dict(img='train/')
#     )
# )
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
    ann_file=data_root + 'annotations/train_sample_10.json',
    metric='bbox',
    classwise=True,
    format_only=False)
test_evaluator = val_evaluator
# test_evaluator = dict(
#     type='CocoMetric',
#     ann_file=data_root + 'annotations/test_sample.json',
#     metric='bbox',
#     format_only=True,
#     outfile_prefix='./outputs/yolox')

# Training settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200, val_interval=1)

# Model settings
model = dict(
    bbox_head=dict(num_classes=_num_classes))

# Runtime settings
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth'
resume = False
checkpoint=dict(type='CheckpointHook', interval=5)
vis_backends = [dict(type='LocalVisBackend'),dict(type='WandbVisBackend'),]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

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