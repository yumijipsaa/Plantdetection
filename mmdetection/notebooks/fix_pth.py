import os
import json

## dataset.json 파일의 img_path에 경로가 잘릴 경우 사용하는 스크립트
## 하단의 사용 예시를 참고하여 수정 후 사용.


def add_prefix_to_file_names(coco_json_path, prefix):
    """
    COCO JSON 파일의 이미지 file_name 앞에 prefix 경로를 붙여 저장합니다.
    """
    print(f"수정 중: {coco_json_path}")
    
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    for image in coco_data["images"]:
        original = image["file_name"]
        image["file_name"] = os.path.normpath(os.path.join(prefix, image["file_name"]))
        print(f'  {original} → {image["file_name"]}')

    with open(coco_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=4)
    print(f"✔ 저장 완료: {coco_json_path}\n")

if __name__ == "__main__":
    # 수정할 대상 JSON 파일과 붙일 경로를 아래에 입력하세요
    add_prefix_to_file_names("data/train/dataset.json", "data/train")
    add_prefix_to_file_names("data/test/dataset.json", "data/test")
    add_prefix_to_file_names("data/val/dataset.json", "data/val")
