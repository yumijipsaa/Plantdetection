_base_ = ["_base_/models/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py",
          "_base_/datasets/tomato_instance.py",
          "_base_/schedules/schedule_1x.py",  
        "_base_/default_runtime.py"]


num_things_classes = 10
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes


data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
    pad_mask=True,
    mask_pad_value=255,
    pad_seg=False,
    # batch_augments=batch_augments
    )

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa
depths = [2, 2, 6, 2] # Swin-Tiny
model = dict(
    data_preprocessor=data_preprocessor,
    type='Mask2Former', 
    backbone=dict(
        _delete_=True, # delete the original backbone
        type='SwinTransformer', # Swin Transformer backbone
        embed_dims=96, # embedding dimension
        depths=depths, # depth of each stage
        num_heads=[3, 6, 12, 24], # number of attention heads in each stage
        window_size=7, # window size for self-attention
        mlp_ratio=4, # ratio of mlp hidden dim to embedding dim
        qkv_bias=True, # whether to use bias in qkv projection
        qk_scale=None, # scale factor for qk projection
        drop_rate=0., # dropout rate for the input
        attn_drop_rate=0., # dropout rate for attention weights
        drop_path_rate=0.3, # stochastic depth rate
        patch_norm=True, # whether to use layer norm after patch embedding
        out_indices=(0, 1, 2, 3), # output from all stages
        with_cp=False, # whether to use checkpointing for memory efficiency
        convert_weights=True, # whether to convert weights to float
        frozen_stages=-1, # freeze all stages
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)), # pretrained weights
    panoptic_head=dict(
        type='Mask2FormerHead',
        in_channels=[96, 192, 384, 768],
        loss_cls=dict(class_weight=[1.0] * num_classes + [0.1]),
        init_cfg=None))

# set all layers in backbone to lr_mult=0.1
# set all norm layers, position_embeding,
# query_embeding, level_embeding to decay_multi=0.0
backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0) # backbone norm layers
backbone_embed_multi = dict(lr_mult=0.1, decay_mult=0.0) # backbone embedding layers
embed_multi = dict(lr_mult=1.0, decay_mult=0.0) # query embedding layers
custom_keys = {
    'backbone': dict(lr_mult=0.1, decay_mult=1.0), # backbone layers
    'backbone.patch_embed.norm': backbone_norm_multi, # patch embedding norm layers
    'backbone.norm': backbone_norm_multi, # backbone norm layers
    'absolute_pos_embed': backbone_embed_multi, # absolute position embedding layers
    'relative_position_bias_table': backbone_embed_multi, # relative position bias table layers
    'query_embed': embed_multi, # query embedding layers
    'query_feat': embed_multi, # query feature layers
    'level_embed': embed_multi # level embedding layers
}
custom_keys.update({
    f'backbone.stages.{stage_id}.blocks.{block_id}.norm': backbone_norm_multi  
    for stage_id, num_blocks in enumerate(depths)
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'backbone.stages.{stage_id}.downsample.norm': backbone_norm_multi
    for stage_id in range(len(depths) - 1)
})
# optimizer
optim_wrapper = dict(
    paramwise_cfg=dict(custom_keys=custom_keys, norm_decay_mult=0.0),
        clip_grad=dict(max_norm=1.0, norm_type=2))