default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,  # epoch마다 저장
        save_best='loss',  # ✅ mAP 대신 loss 기준으로 best model 저장
        rule='less'  # ✅ loss 기준으로 더 낮은 값이 더 좋음
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    early_stopping=dict(
        type='EarlyStoppingHook', 
        monitor='loss',  # ✅ loss 기준으로 early stopping
        rule='less',
        patience=200,  # 200 epoch 동안 loss 개선 없으면 종료
        min_delta=0.001
    )
)


# 환경 설정
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl')
)

# Log Processor
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

# 기타
log_level = 'INFO'
load_from = None
resume = False
