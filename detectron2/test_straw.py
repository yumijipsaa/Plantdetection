import os
import json
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

# 🔹 데이터셋 정보
dataset_name = "straw_test"
json_path = "D:/plantdetection/detectron2/data/straw_test/dataset.json"
image_root = "D:/plantdetection/detectron2"  
output_folder = "D:/plantdetection/detectron2/inference_result/"

# 🔹 출력 폴더 생성
os.makedirs(output_folder, exist_ok=True)

# 🔹 COCO 데이터셋 등록
register_coco_instances(dataset_name, {}, json_path, image_root)


# class_names = ['flower', 'fruit', 'cap', 'leaf', 'midrib', 'stem', 'node', 'cap_2']

# 🔹 모델 설정 불러오기
cfg = get_cfg()
cfg.merge_from_file("model_zoo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")


cfg.MODEL.WEIGHTS = "2024-05-20_1419_strawberry_208749.pth"  # 학습된 가중치
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6  # 클래스 개수 고정
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1  # 신뢰도 임계값
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # GPU 사용 가능하면 GPU 사용



# # 🔹 모델 설정 불러오기
# cfg = get_cfg()
# cfg.merge_from_file("model_zoo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
# cfg.MODEL.WEIGHTS = "2024-04-16_1039_paprika_658000.pth"  # 학습된 가중치
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_names)
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1  # 신뢰도 임계값 낮춰서 분석
# cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 🔹 Windows multiprocessing 문제 방지
if __name__ == '__main__':
    # 🔹 mAP 평가
    evaluator = COCOEvaluator(dataset_name, cfg, False, output_dir=output_folder)
    val_loader = build_detection_test_loader(cfg, dataset_name, num_workers=0)
    model = DefaultPredictor(cfg).model

    print("🔍 평가 중...")
    metrics = inference_on_dataset(model, val_loader, evaluator)

    # 🔹 mAP 결과 저장
    map_result_path = os.path.join(output_folder, "map_results.json")
    with open(map_result_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\n✅ mAP 결과 저장 완료: {map_result_path}")
    print(f"📊 mAP 결과: {metrics}")
