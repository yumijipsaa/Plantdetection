default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),  # 반복 시간 측정
    logger=dict(type='LoggerHook', interval=50),  # 로그 출력 주기 (50번마다)
    param_scheduler=dict(type='ParamSchedulerHook'),  # 학습률 스케줄러 제어
    checkpoint=dict(type='CheckpointHook', interval=1),  # 매 epoch마다 체크포인트 저장
    sampler_seed=dict(type='DistSamplerSeedHook'),  # 분산 학습 시 시드 설정
    visualization=dict(type='DetVisualizationHook')  # 시각화 훅 (DetLocalVisualizer와 연결됨)
)

env_cfg = dict(
    cudnn_benchmark=False,  # CuDNN에서 input 크기 고정 여부
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),  # 멀티프로세싱 설정
    dist_cfg=dict(backend='nccl')  # 분산 학습 backend 설정 (NVIDIA 환경)
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False
