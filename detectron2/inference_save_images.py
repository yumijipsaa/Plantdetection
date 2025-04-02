import os
import cv2
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer

# 🔹 입력 이미지 폴더 및 출력 폴더 설정
image_folder = "D:/plantdetection/mmdetection/data/melon_test/"
output_folder = "D:/plantdetection/detectron2/inference_result/"

# 🔹 출력 폴더 생성 (없으면 생성)
os.makedirs(output_folder, exist_ok=True)

# 🔹 모델 설정 불러오기
cfg = get_cfg()
cfg.merge_from_file("model_zoo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = "2022-06-23_123_melon_500000.pth"  # 학습된 가중치
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7  # 🔹 커스텀 데이터셋 클래스 수 맞추기
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 신뢰도 임계값
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # GPU 사용 가능하면 GPU 사용

# 🔹 모델 로드
predictor = DefaultPredictor(cfg)

# 🔹 이미지 폴더 내 모든 이미지 처리
for filename in os.listdir(image_folder):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):  # 이미지 파일만 처리
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"⚠️ 이미지 로드 실패: {img_path}")
            continue  # 다음 이미지로 넘어가기

        # 🔹 예측 수행
        predictions = predictor(img)

        # 🔹 시각화 및 결과 저장
        visualizer = Visualizer(img[:, :, ::-1], MetadataCatalog.get("melon_test"), scale=0.5)
        out = visualizer.draw_instance_predictions(predictions["instances"].to("cpu"))

        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, out.get_image()[:, :, ::-1])  # 결과 저장

        print(f"✅ 저장 완료: {output_path}")

print("🎉 모든 이미지 처리 완료!")
