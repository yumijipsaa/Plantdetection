from mmengine import Config

cfg = Config.fromfile('custom_configs/casecade-mask-rcnn_r50_fpn_melon.py')

# `_base_`가 정상적으로 로드되었는지 확인
print("Base configs:", cfg.get('_base_', 'No _base_ found'))
