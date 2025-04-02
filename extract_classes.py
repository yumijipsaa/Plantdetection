### coco dataset의 class를 추출하여 json 파일로 추출한다.

import os
import json
from pycocotools.coco import COCO

### coco data 경로 입력
coco_path = os.path.join("mmdetection/data/train/tomato_seed.json")
save_path = os.path.join("for_inference", "2024-11-28_tomato_seed.json")

def load_coco_data(coco_path):
    coco = COCO(coco_path)
    cats = coco.loadCats(coco.getCatIds())
    classes = [cat['name'] for cat in cats]
    return classes

if __name__ == "__main__":
    classes = load_coco_data(coco_path=coco_path)
    with open(save_path, "w") as f:
        json.dump(classes, f)
    print(classes)