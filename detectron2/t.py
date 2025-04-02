# import json
# from collections import Counter

# # COCO 형식 annotation 파일 경로
# json_path = "data/pa_test/dataset.json"

# with open(json_path, 'r') as f:
#     data = json.load(f)

# # 각 annotation의 category_id 수집
# category_counts = Counter(ann['category_id'] for ann in data['annotations'])

# # category_id -> name 매핑
# id_to_name = {cat['id']: cat['name'] for cat in data['categories']}

# # 결과 출력
# print("📊 클래스 분포:")
# for cid, count in sorted(category_counts.items()):
#     name = id_to_name.get(cid, "Unknown")
#     print(f" - {name:10}: {count}개")









import cv2
import os
import json
import torch
import matplotlib.pyplot as plt
from collections import defaultdict

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode, Visualizer

# 📁 데이터셋 정보
dataset_name = "pa_test"
json_path = "D:/plantdetection/detectron2/data/pa_test/dataset.json"
image_root = "D:/plantdetection/detectron2"
class_names = ['flower', 'fruit', 'cap', 'leaf', 'midrib', 'stem', 'node', 'cap_2']

# 📦 데이터셋 등록
register_coco_instances(dataset_name, {}, json_path, image_root)
MetadataCatalog.get(dataset_name).thing_classes = class_names

# ⚙️ 설정
cfg = get_cfg()
cfg.merge_from_file("model_zoo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = "2024-04-16_1039_paprika_658000.pth"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0  # 모든 예측을 받기 위해 낮게 설정
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

predictor = DefaultPredictor(cfg)

# 📊 클래스별 confidence 저장 딕셔너리
confidence_per_class = defaultdict(list)

# 🔍 예측 수행
dataset_dicts = DatasetCatalog.get(dataset_name)

print("🔍 예측 중...")

for i, d in enumerate(dataset_dicts):
    image = cv2.imread(d["file_name"])
    outputs = predictor(image)
    instances = outputs["instances"]
    if len(instances) == 0:
        continue

    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    for cls_id, score in zip(classes, scores):
        class_name = class_names[cls_id]
        confidence_per_class[class_name].append(score)

print("✅ 예측 완료!")

# 📈 히스토그램 시각화
plt.figure(figsize=(16, 6))
for idx, class_name in enumerate(class_names):
    plt.subplot(2, 4, idx + 1)
    scores = confidence_per_class[class_name]
    if scores:
        plt.hist(scores, bins=20, range=(0, 1), alpha=0.75)
    plt.title(f"{class_name} ({len(scores)} preds)")
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.grid(True)

plt.tight_layout()
plt.suptitle("🔍 Class-wise Confidence Score Histogram", fontsize=16, y=1.05)
plt.show()
