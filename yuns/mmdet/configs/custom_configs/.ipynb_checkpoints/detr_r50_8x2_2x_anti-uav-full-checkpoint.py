_base_ = '../detr/detr_r50_8xb2-150e_coco.py'
model = dict(bbox_head=dict(num_classes=1,))
classes = ('drone',)
checkpoint_config = dict(interval=3)

data = dict(
    samples_per_gpu=3,
    workers_per_gpu=4,
    train=dict(
        img_prefix='/home/ubuntu/workspace/datasets/eo_videos_to_image_new/',
        classes=classes,
        ann_file='/home/ubuntu/workspace/datasets/annotations/eo_test.json'),
    val=dict(
        img_prefix='/home/ubuntu/workspace/datasets/eo_videos_to_image_new/',
        classes=classes,
        ann_file='/home/ubuntu/workspace/datasets/annotations/eo_test.json'),
    test=dict(
        img_prefix='/home/ubuntu/workspace/datasets/eo_videos_to_image_new/',
        classes=classes,
        ann_file=''))

load_from = '/home/ubuntu/workspace/yuns/models/detr_r50_8x2_2x_anti-uav-full_model.pth'
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
total_epochs = 24
