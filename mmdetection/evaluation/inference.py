import argparse
import os
import mmcv
import json
from mmdet.apis import DetInferencer
from mmengine.config import Config
from mmengine.visualization import LocalVisBackend


# 예시)
# python evaluation/inference.py --config custom_configs/cascade-mask-rcnn_r50_fpn_onion.py 
# --ckpt work_dirs/epoch_218.pth --data_dir evaluation_results/inference_result/onion_test



def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection Inference Script')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--image', type=str, help='Path to a single image for inference')
    parser.add_argument('--data_dir', type=str, help='Path to a folder of images for inference')
    parser.add_argument('--work_dirs', type=str, default="work_dirs", help="Work directory path")
    return parser.parse_args()

def main():
    args = parse_args()

    # 파일 검증
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint file not found: {args.ckpt}")

    # 모델 초기화
    inferencer = DetInferencer(model=args.config, weights=args.ckpt, device='cpu')


    # 결과 저장 디렉토리
    save_dir = "inference_result/"
    os.makedirs(save_dir, exist_ok=True)

    # 단일 이미지 처리
    if args.image:
        if not os.path.exists(args.image):
            raise FileNotFoundError(f"Image file not found: {args.image}")
        
        img = mmcv.imread(args.image)
        result = inferencer(img, return_vis=True, return_datasamples=True)

        output_path = os.path.join(save_dir, os.path.basename(args.image))
        mmcv.imwrite(result['visualization'][0], output_path)
        print(f"Saved result to {output_path}")

    # 폴더 내 이미지 처리
    elif args.data_dir:
        if not os.path.exists(args.data_dir):
            raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
        
        print(f"Processing images in: {args.data_dir}")
        for img_name in os.listdir(args.data_dir):
            if not img_name.endswith('.jpg'):
                print(f'Skipping non-JPG file: {img_name}')
                continue

            img_path = os.path.join(args.data_dir, img_name)
            img = mmcv.imread(img_path)
            result = inferencer(img, return_vis=True, return_datasamples=True)
            output_path = os.path.join(save_dir, img_name)
            mmcv.imwrite(result['visualization'][0], output_path)
            print(f"Saved result to {output_path}")

    else:
        print("Error: You must provide either --image or --data_dir")


if __name__ == '__main__':
    main()
