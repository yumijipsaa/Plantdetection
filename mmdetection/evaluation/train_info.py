import re
import torch
from mmengine import Config
from collections import OrderedDict

# info가 필요한 로그 파일 경로
#log_file = 'work_dirs/casecade-mask-rcnn_r50_fpn_melon/20250305_160537/20250305_160537.log'
log_file = 'work_dirs/cascade-mask-rcnn_r50_fpn_onion/20250320_180829.log'

# 결과 저장할 딕셔너리
log_data = {
    "total_loss": "N/A",
    "Training Time": "N/A",
    "GPU Model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    "loss_rpn_cls": "N/A",
    "epoch": "N/A",
    "learning rate": "N/A",
    "batch size": "N/A",
    "score_threshold": "N/A",
    "Dataset": "N/A",
    "Optimizer": "N/A"
}


# 🔹 1️⃣ Training Time 계산을 위한 변수 추가
total_training_time = None
num_iterations_per_epoch = 48  # 1 epoch당 iteration 개수 (로그에서 확인 필요)

# 🔹 1️⃣ 로그 파일에서 추출 가능한 값 찾기
with open(log_file, 'r') as f:
    logs = f.readlines()

# 첫 번째로 등장하는 날짜 찾기 (YYYY/MM/DD 형식)
for line in logs:
    match = re.search(r'(\d{4}/\d{2}/\d{2})', line)
    if match:
        log_data["Date"] = match.group(1)
          
    if "loss:" in line:
        match = re.search(r'loss:\s([\d.]+)', line)
        if match:
            log_data["total_loss"] = float(match.group(1))

    # 🔥 `Training Time` 찾기 (가장 마지막 `time:` 값을 사용)
    if "time:" in line:
        match = re.search(r'time:\s([\d.]+)', line)
        if match:
            iteration_time = float(match.group(1))
            log_data["Training Time"] = f"{iteration_time} sec per iteration"

            # 🔥 총 학습 시간 계산 (epoch 개수와 iteration 개수 활용)
            if "epoch" in log_data and log_data["epoch"] != "N/A":
                total_training_time = iteration_time * log_data["epoch"] * num_iterations_per_epoch

    if "loss_rpn_cls:" in line:
        match = re.search(r'loss_rpn_cls:\s([\d.]+)', line)
        if match:
            log_data["loss_rpn_cls"] = float(match.group(1))

    if "Epoch(train)" in line:
        match = re.search(r'Epoch\(train\)\s+\[(\d+)', line)
        if match:
            log_data["epoch"] = int(match.group(1))

    if "lr:" in line:
        match = re.search(r'lr:\s([\d.e-]+)', line)
        if match:
            log_data["learning rate"] = float(match.group(1))



if total_training_time:
    hours = int(total_training_time // 3600)
    minutes = int((total_training_time % 3600) // 60)
    seconds = int(total_training_time % 60)
    log_data["Total Training Time"] = f"{hours}h {minutes}m {seconds}s"

# 🔹 2️⃣ `config.py`에서 `score_threshold` 가져오기
cfg = Config.fromfile('custom_configs/cascade-mask-rcnn_r50_fpn_onion.py')

if hasattr(cfg.model, 'test_cfg') and hasattr(cfg.model.test_cfg, 'rcnn'):
    log_data["score_threshold"] = cfg.model.test_cfg.rcnn.get('score_thr', "N/A")

# 🔹 3️⃣ `melon_instance.py` 직접 불러와서 `Dataset`, `batch_size`, `Optimizer` 가져오기
data_cfg = Config.fromfile('custom_configs/_base_/datasets/onion_instance.py')

# Dataset 가져오기
if hasattr(cfg, 'dataset_type'):
    log_data["Dataset"] = data_cfg.dataset_type

# Batch Size 가져오기
if hasattr(data_cfg, 'train_dataloader') and 'batch_size' in data_cfg.train_dataloader:
    log_data["batch size"] = data_cfg.train_dataloader.batch_size

# `optim_wrapper` 안에서 `optimizer` 가져오기
if hasattr(cfg, 'optim_wrapper') and hasattr(cfg.optim_wrapper, 'optimizer'):
    log_data["Optimizer"] = cfg.optim_wrapper.optimizer.get('type', "N/A")

# 🔹 1️⃣ 사용한 모델과 백본 찾기
if hasattr(cfg, 'model'):
    log_data["Model"] = cfg.model.get("type", "N/A")
    
    if hasattr(cfg.model, 'backbone'):
        backbone_type = cfg.model.backbone.get("type", "N/A")
        backbone_depth = cfg.model.backbone.get("depth", "")  # Depth 가져오기
        log_data["Backbone"] = f"{backbone_type} {backbone_depth}".strip()


# `Date` → `Model` → `Backbone` 순서로 정렬
ordered_log_data = OrderedDict([
    ("Date", log_data["Date"]),
    ("Model", log_data["Model"]),
    ("Backbone", log_data["Backbone"]),
    ("score_threshold", log_data["score_threshold"]),
    ("learning rate", log_data["learning rate"]),
    ("batch size", log_data["batch size"]),
    ("epoch", log_data["epoch"]),
    ("total_loss", log_data["total_loss"]),
    ("Total Training Time", log_data.get("Total Training Time", "N/A")),  # 🔥 총 학습 시간 추가
    ("GPU Model", log_data["GPU Model"]),
    ("loss_rpn_cls", log_data["loss_rpn_cls"]),
    ("Dataset", log_data["Dataset"]),
    ("Optimizer", log_data["Optimizer"])
])

# 🔹 4️⃣ 최종 정렬된 결과 출력
print("\n🔹 Extracted Training Information 🔹")
for key, value in ordered_log_data.items():
    print(f"{key}: {value}")


log_file_path = "evaluation_results/train_info/training_log.txt"

# 파일 저장
with open(log_file_path, "w", encoding="utf-8") as f:
    for key, value in ordered_log_data.items():
        f.write(f"{key}: {value}\n")

print(f"로그 파일이 저장되었습니다: {log_file_path}")



