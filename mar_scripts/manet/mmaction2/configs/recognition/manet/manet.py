_base_ = [
    '../../_base_/models/manet_r50.py', '../../_base_/schedules/sgd_manet_80e.py',
    '../../_base_/default_runtime.py'
]

# dataset settings
dataset_type = 'VideoDataset'
data_root = '/data/stars/share/MA_52/train'
data_root_val = '/data/stars/share/MA_52/val/'
data_root_test = './data/ma52/videos_test/'
ann_file_train = '/data/stars/share/MA_52/annotations/annotations_train_list_videos.txt'
ann_file_val = '/data/stars/share/MA_52/annotations/annotations_val_list_videos.txt'
ann_file_test = './data/ma52/test_list_videos.txt'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_bgr=False)

train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', 
         clip_len=1,
         frame_interval=1,
         num_clips=8),
    dict(type='DecordDecode'),
    dict(type='Resize', 
         scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1,
        num_fixed_crops=13),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label','emb'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label','emb'])
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label','emb'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label','emb'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label','emb'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label','emb'])
]
data = dict(
    videos_per_gpu=10,
    workers_per_gpu=4,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_test,
        pipeline=test_pipeline))
evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
optimizer = dict(
    lr=0.01/8, 
)
# runtime settings
checkpoint_config = dict(interval=5)
work_dir = './work_dirs/manet_videomaev2_extracted_features_no_addition/'
