# 이미지와 JSON 라벨 파일을 설정된 비율에 따라 train, val, test 폴더로 분할하는 스크립트이다.
# split_images() 함수에 분할 대상 폴더 경로를 넣어 실행한다.
# 분할 비율은 ratio 인자를 수정하여 조정할 수 있다.

import os
import random
import shutil
from pathlib import Path

def split_images(
    image_dir,
    label_dir,
    output_dir,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42
):
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 이미지 파일 목록 (jpg, png 등)
    image_files = [f for f in image_dir.glob('*.*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    image_files = sorted(image_files)
    total = len(image_files)

    random.seed(seed)
    random.shuffle(image_files)

    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)

    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]

    def copy_pair(files, subset_name):
        subset_path = output_dir / subset_name
        subset_path.mkdir(parents=True, exist_ok=True)

        for img_file in files:
            base_name = img_file.stem
            json_file = label_dir / f"{base_name}.json"
            # 이미지 복사
            shutil.copy(img_file, subset_path / img_file.name)
            # JSON 복사
            if json_file.exists():
                shutil.copy(json_file, subset_path / json_file.name)
            else:
                print(f"⚠️ {json_file.name} (JSON) 누락됨")

    copy_pair(train_files, 'train')
    copy_pair(val_files, 'val')
    copy_pair(test_files, 'test')

    print(f"✅ 분할 완료! 총 {total}개:")
    print(f" - train: {len(train_files)}개")
    print(f" - val:   {len(val_files)}개")
    print(f" - test:  {len(test_files)}개")
    print(f"→ 저장 위치: {output_dir.resolve()}")

#사용 예시:
split_images(
    image_dir='data/tomato',
    label_dir='data/tomato',
    output_dir='data'
)
