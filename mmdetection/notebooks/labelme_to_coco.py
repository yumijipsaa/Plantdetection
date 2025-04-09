import labelme2coco
import copy

# 데이터셋 경로
train_labelme_dir = "data/train/"
test_labelme_dir = "data/test/"
val_labelme_dir = "data/val/"

# COCO 데이터셋 저장 경로
coco_train_json = "data/train/dataset.json"
coco_test_json = "data/test/dataset.json"
coco_val_json = "data/val/dataset.json"

# 개별 실행을 위한 임시 변수
train_coco_data = None
test_coco_data = None
val_coco_data = None

# 학습 데이터 COCO 변환 (deepcopy 사용하여 충돌 방지)
train_coco_data = labelme2coco.convert(train_labelme_dir, "data/train/")
train_coco_data = copy.deepcopy(train_coco_data)  # ✅ 기존 데이터 유지

# # 테스트 데이터 COCO 변환 (새로운 객체로 처리)
# test_coco_data = labelme2coco.convert(test_labelme_dir, "data/test/")
# test_coco_data = copy.deepcopy(test_coco_data)  # ✅ 기존 데이터 유지

# # 검증증 데이터 COCO 변환 (새로운 객체로 처리)
# val_coco_data = labelme2coco.convert(val_labelme_dir, "data/val/")
# val_coco_data = copy.deepcopy(val_coco_data)  # ✅ 기존 데이터 유지

