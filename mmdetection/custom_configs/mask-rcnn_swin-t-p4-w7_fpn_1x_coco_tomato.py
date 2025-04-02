_base_ = [
    "_base_/models/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py",
    "_base_/datasets/tomato_instance.py",
    "_base_/default_runtime.py",
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.05))