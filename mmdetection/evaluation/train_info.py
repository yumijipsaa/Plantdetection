import re
import argparse
from mmengine import Config
from collections import OrderedDict
import torch

### 이 스크립트는 MMDetection 학습 로그 파일과 config 파일을 분석해,
### 모델 정보, 학습 설정, 총 학습 시간, GPU 정보 등을 요약 출력해주는 도구입니다.

### 사용 예시)
### python evaluation/train_info.py --log work_dirs/20250403_133443.log --config custom_configs/mask2former_r50_8xb2-lsj-50e_coco_tomato.py
### 로그와 config 파일 경로를 인자로 받아서 실행합니다.

def extract_info(log_path, config_path):
    cfg = Config.fromfile(config_path)

    log_data = OrderedDict({
        "Date": "N/A",
        "Model": cfg.model.get("type", "N/A"),
        "Backbone": f"{cfg.model.backbone.get('type', '')} {cfg.model.backbone.get('depth', '')}".strip(),

        # ✅ 다양한 모델에서 사용할 수 있도록 RCNN이 아닌 공통 필드로 score_threshold 확인
        "score_threshold": cfg.model.get("test_cfg", {}).get("score_thr", "N/A"),

        "learning rate": cfg.get("optim_wrapper", {}).get("optimizer", {}).get("lr", "N/A"),
        "batch size": cfg.get("train_dataloader", {}).get("batch_size", "N/A"),
        "epoch": "N/A",
        "total_loss": "N/A",
        "Total Training Time": "N/A",
        "GPU Model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",

        # ✅ Mask2Former 등 RPN이 없는 모델에서는 로그에 안 나올 수 있으므로 N/A로 표시
        "loss_rpn_cls": "N/A",

        "Dataset": cfg.get("dataset_type", "N/A"),
        "Optimizer": cfg.get("optim_wrapper", {}).get("optimizer", {}).get("type", "N/A")
    })

    with open(log_path, 'r', encoding='latin1') as f:
        logs = f.readlines()

    last_time = None
    num_iterations_per_epoch = 48

    for line in logs:
        if log_data["Date"] == "N/A":
            date_match = re.search(r'(\d{4}/\d{2}/\d{2})', line)
            if date_match:
                log_data["Date"] = date_match.group(1)

        if "Epoch(train)" in line:
            match = re.search(r'Epoch\(train\)\s+\[(\d+)', line)
            if match:
                log_data["epoch"] = int(match.group(1))

        if "loss:" in line and log_data["total_loss"] == "N/A":
            match = re.search(r'loss:\s([\d.]+)', line)
            if match:
                log_data["total_loss"] = float(match.group(1))

        # ✅ RPN이 있는 모델에서만 출력되므로, 없을 경우 그대로 N/A 유지됨
        if "loss_rpn_cls:" in line and log_data["loss_rpn_cls"] == "N/A":
            match = re.search(r'loss_rpn_cls:\s([\d.]+)', line)
            if match:
                log_data["loss_rpn_cls"] = float(match.group(1))

        if "time:" in line:
            match = re.search(r'time:\s([\d.]+)', line)
            if match:
                last_time = float(match.group(1))

    if last_time and log_data["epoch"] != "N/A":
        total_sec = last_time * int(log_data["epoch"]) * num_iterations_per_epoch
        hours = int(total_sec // 3600)
        minutes = int((total_sec % 3600) // 60)
        seconds = int(total_sec % 60)
        log_data["Total Training Time"] = f"{hours}h {minutes}m {seconds}s"

    print("\n🔹 Extracted Training Information 🔹")
    for key, value in log_data.items():
        print(f"{key}: {value}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', required=True, help='Path to training log file')
    parser.add_argument('--config', required=True, help='Path to training config file')
    args = parser.parse_args()
    extract_info(args.log, args.config)
