# MMDetection에서 사용하는 데이터 설정(config) 파일로, 학습·검증·테스트에 사용할 데이터셋, 파이프라인, 평가 방식을 정의


dataset_type = 'CocoDataset' # 사용할 데이터셋 형식 (COCO 포맷)
data_root = 'data/' # 데이터셋 루트 디렉토리
backend_args = None # 파일 시스템 백엔드 설정 (로컬 파일이므로 None)


metainfo = {
    'classes' : ("class1","class2",)} # 클래스 이름 리스트 (dataset.json의 구성요소와 값이 맞아야한다.)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args), # 이미지 파일 로딩
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True), # 어노테이션 로딩 (box + mask)
    dict(type='Resize', scale=(1024,1024), keep_ratio=False), # 이미지 크기 고정 (비율 유지 X)
    dict(type='RandomFlip', prob=0.5), # 수평 뒤집기 확률 50%
    dict(type='RandomRotate', prob=0.5, angle_range=(-30, 30)),  # 📌 추가: 랜덤 회전 (각도 범위 ±30도)
    dict(type='PackDetInputs') # 모델 입력 형태로 데이터 포장
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args), # 테스트 이미지 로딩
    dict(type='Resize', scale=(1024, 1024), keep_ratio=False), # 테스트 이미지 크기 고정
    # If you don't have a gt annotation, delete the pipeline
    # dict(
    #     type='LoadAnnotations', with_bbox=True, with_mask=True, with_seg=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor')) # 평가 시 메타 정보 저장
]

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True, # 워커를 epoch 간 유지
    sampler=dict(type='DefaultSampler', shuffle=True), # 기본 샘플링, 무작위 순서
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_prefix=dict(img=''), # 이미지 경로 prefix (data_root 기준 상대경로)
        ann_file= data_root + 'train/dataset.json', # 학습용 dataset.json 경로
        metainfo = metainfo, # 클래스 정보
        filter_cfg=dict(filter_empty_gt=True, min_size=32), # 빈 GT 제거, 최소 크기 필터링
        pipeline=train_pipeline, # 학습용 파이프라인 지정
        backend_args=backend_args)) # 백엔드 설정

test_dataloader = dict(
    batch_size=1, # 테스트 시 한 번에 1장
    num_workers=2,
    persistent_workers=True,
    drop_last=False, # 마지막 배치를 버리지 않음
    sampler=dict(type='DefaultSampler', shuffle=False), # 순차적 순서 유지
    dataset=dict(
        type=dataset_type,
        metainfo = metainfo,
        ann_file= data_root + 'test/dataset.json',
        data_prefix=dict(img=''),
        test_mode=True, # 테스트 모드 (학습 X, 평가용)
        pipeline=test_pipeline,
        backend_args=backend_args))


test_evaluator = dict(
    type='CocoMetric', # COCO 방식 평가 지표 사용
    ann_file= data_root + 'test/dataset.json',
    metric=['bbox', 'segm'],# 박스 및 세그멘테이션 평가
    format_only=False, # 포맷만 저장할 건지 여부 (False이면 평가까지 수행)
    backend_args=backend_args)


val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),# GT 포함
    dict(type='Resize', scale=(1024, 1024), keep_ratio=False), # 크기 고정
    # dict(type='RandomFlip', prob=0.5), # 주석 처리: 검증은 보통 augment 없이 진행
    dict(type='PackDetInputs')
]

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type, 
        ann_file= data_root + 'val/dataset.json',
        metainfo = metainfo,
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=val_pipeline,
        backend_args=backend_args))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val/dataset.json',
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=backend_args)

val_cfg = dict() # 사용하지 않는 검증 설정을 위한 placeholder (예비용)

