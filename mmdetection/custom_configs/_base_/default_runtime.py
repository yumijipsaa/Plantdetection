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

vis_backends = [dict(type='LocalVisBackend')] # 시각화 백엔드 설정 (로컬)
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer') # 시각화기 설정 (DetLocalVisualizer와 연결됨)
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True) # 로그 프로세서 설정 (50번마다 로그 처리)

log_level = 'INFO' # 로그 레벨 설정 (INFO)
load_from = None # Pre-trained 모델 경로 (없음)
resume = False # 학습 재개 여부 (False)
