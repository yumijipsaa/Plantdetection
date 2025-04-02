_base_ = [
    "_base_/models/mask-rcnn_r50_fpn.py",
    "_base_/datasets/tomato_instance.py",
    "_base_/schedules/custom_scheduler.py",
    "_base_/default_runtime.py"
]