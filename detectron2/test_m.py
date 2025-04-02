import os
import torch
import cv2
import glob
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# 🔹 데이터셋 정보
dataset_name = "m_test"
json_path = "D:/plantdetection/detectron2/data/m_test/dataset.json"
image_root = "D:/plantdetection/detectron2"  
output_folder = "D:/plantdetection/detectron2/inference_result/"

# 🔹 출력 폴더 생성
os.makedirs(output_folder, exist_ok=True)

# 🔹 COCO 데이터셋 등록
register_coco_instances(dataset_name, {}, json_path, image_root)

# 🔹 모델 설정 불러오기
cfg = get_cfg()
cfg.merge_from_file("model_zoo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = "2022-06-23_123_melon_500000.pth"  # 학습된 가중치
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7  # 클래스 개수 고정
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1  # 신뢰도 임계값
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # GPU 사용 가능하면 GPU 사용

# 🔹 Windows multiprocessing 문제 방지
if __name__ == '__main__':
    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get(dataset_name)

    image_paths = glob.glob(os.path.join(image_root, "data/m_test", "*.jpg"))
    if not image_paths:
        print("❌ 추론할 이미지가 없습니다.")
        exit()

    for path in image_paths:
        img = cv2.imread(path)
        outputs = predictor(img)

        v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
        vis_img = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()[:, :, ::-1]

        cv2.imshow("Inference Result", vis_img)
        print(f"📂 파일: {os.path.basename(path)}")
        key = cv2.waitKey(0)
        if key == 27:  # ESC 입력 시 종료
            break

    cv2.destroyAllWindows()
