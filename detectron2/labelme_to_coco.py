import labelme2coco
import copy


test_labelme_dir = "data/m_test/"


coco_test_json = "data/m_test/dataset.json"

test_coco_data = None



# 테스트 데이터 COCO 변환 (새로운 객체로 처리)
test_coco_data = labelme2coco.convert(test_labelme_dir, "data/m_test/")
test_coco_data = copy.deepcopy(test_coco_data)  # ✅ 기존 데이터 유지


