dataset_type = 'CocoDataset'
data_root = 'data/'
backend_args = None

metainfo = {
    'classes' : (
         "leaf", 
         "stem", 
         "top_fruit", 
         "fruit"
         ),
}
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomRotate', prob=0.5, angle_range=(-30, 30)),  # 📌 추가: 랜덤 회전
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    #dict(type='LoadAnnotations', with_bbox=True, with_mask=True, with_seg=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        ann_file= data_root + 'onion/dataset.json',
        metainfo = metainfo,
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file= data_root + 'onion_test/dataset.json',
        metainfo = metainfo,
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))


test_evaluator = dict(
    type='CocoMetric',
    ann_file= data_root + 'onion_test/dataset.json',
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=backend_args)



# val_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=backend_args),
#     dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
#     dict(type='Resize', scale=(1333, 800), keep_ratio=True),   
#     dict(type='PackDetInputs')
# ]

# val_dataloader = dict(
#     batch_size=2,
#     num_workers=2,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     batch_sampler=dict(type='AspectRatioBatchSampler'),
#     dataset=dict(
#         type=dataset_type,
        
#         ann_file= data_root + 'onion_test/dataset.json',
#         metainfo = metainfo,
#         data_prefix=dict(img=''),
#         filter_cfg=dict(filter_empty_gt=True, min_size=32),
#         pipeline=val_pipeline,
#         backend_args=backend_args))

# val_evaluator = dict(
#     type='CocoMetric',
#     ann_file=data_root + 'onion_test/dataset.json',
#     metric=['bbox', 'segm'],
#     format_only=False,
#     backend_args=backend_args)




