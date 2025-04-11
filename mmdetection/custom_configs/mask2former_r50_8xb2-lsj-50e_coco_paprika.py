# train.py에 지정되는 config 파일로 _base_/models 경로를 지정해 기본 설정값을 불러오고
# 최종적으로는 이 파일의 설정을 덮어씌워 사용한다.
# 기본 설정 -> base models 불러옴 최종 커스텀 설정 : 이 파일 사용


_base_ = ["_base_/models/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py",
          "_base_/datasets/paprika_instance.py",
        "_base_/schedules/schedule_1x.py", 
        "_base_/default_runtime.py"]



num_things_classes = 7
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes
# image_size = (1024, 1024)
# batch_augments = [
#     dict(
#         type='BatchFixedSizePad',
#         # size=image_size,
#         img_pad_value=0,
#         pad_mask=True,
#         mask_pad_value=0,
#         pad_seg=False)
# ]
data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
    pad_mask=True,
    mask_pad_value=0,
    pad_seg=False,
    # batch_augments=batch_augments
    )
model = dict(
    data_preprocessor=data_preprocessor,
    panoptic_head=dict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_cls=dict(class_weight=[1.0] * num_classes + [0.1])),
        

    panoptic_fusion_head=dict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes),
    test_cfg=dict(panoptic_on=False,
                  instance_on=True),


    train_cfg=dict(
        num_points=12544,  # 일반적으로 112x112 (512 크기 이미지 기준)
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='ClassificationCost', weight=2.0),
                dict(type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
                dict(type='DiceCost', weight=5.0, pred_act=True, eps=1.0)
            ]
        ),
        sampler=dict(type='MaskPseudoSampler')
    )
)

# dataset settings
# train_pipeline = [
#     dict(
#         type='LoadImageFromFile',
#         to_float32=True,
#         backend_args={{_base_.backend_args}}),
#     dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
#     dict(type='RandomFlip', prob=0.5),
#     # large scale jittering
#     dict(
#         type='RandomResize',
#         scale=image_size,
#         ratio_range=(0.1, 2.0),
#         resize_type='Resize',
#         keep_ratio=True),
#     dict(
#         type='RandomCrop',
#         crop_size=image_size,
#         crop_type='absolute',
#         recompute_bbox=True,
#         allow_negative_crop=True),
#     dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
#     dict(type='PackDetInputs')
# ]

# test_pipeline = [
#     dict(
#         type='LoadImageFromFile',
#         to_float32=True,
#         backend_args={{_base_.backend_args}}),
#     dict(type='Resize', scale=(1333, 800), keep_ratio=True),
#     # If you don't have a gt annotation, delete the pipeline
#     dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
#     dict(
#         type='PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
#                    'scale_factor'))
# ]

# dataset_type = 'CocoDataset'
# data_root = 'data/coco/'

# train_dataloader = dict(
#     dataset=dict(
#         type=dataset_type,
#         ann_file='annotations/instances_train2017.json',
#         data_prefix=dict(img='train2017/'),
#         pipeline=train_pipeline))
# val_dataloader = dict(
#     dataset=dict(
#         type=dataset_type,
#         ann_file='annotations/instances_val2017.json',
#         data_prefix=dict(img='val2017/'),
#         pipeline=test_pipeline))
# test_dataloader = val_dataloader

# val_evaluator = dict(
#     _delete_=True,
#     type='CocoMetric',
#     ann_file=data_root + 'annotations/instances_val2017.json',
#     metric=['bbox', 'segm'],
#     format_only=False,
#     backend_args={{_base_.backend_args}})
# test_evaluator = val_evaluator
