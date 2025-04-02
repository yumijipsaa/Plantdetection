import argparse
import os
import cv2
import mmcv
import numpy as np
import json

from mmdet.apis import DetInferencer
from mmengine.config import Config
from mmengine.visualization import LocalVisBackend

from mmdet.apis import init_detector, inference_detector

ckpt_path = 'work_dirs/mask2former_r50_tomato_all_data/epoch_125.pth'
config_path = 'custom_configs/mask2former_r50_8xb2-lsj-50e_coco_tomato.py'
test_data_dir = 'data/test'

def main():
    # inferencer 설정
    inferencer = DetInferencer(model=config_path, weights=ckpt_path, device='cpu')
    # meta info
    classes = inferencer.model.cfg['metainfo']['classes']
    print(f"classes : {classes}")
    
    for test_img in os.listdir(test_data_dir)[25:]:
        img = mmcv.imread(os.path.join(test_data_dir, test_img))
        result = inferencer(img, show=False, return_datasamples=True, return_vis=True)
        # mmcv.imwrite(result['visualization'][0], os.path.join("test_img", test_img.replace(".jpg", "_inference.jpg")))
        h,w,_ = img.shape
        mask_img = img.copy()
        mask_img2 = img.copy()
        for i in range(len(result['predictions'][0].pred_instances.labels)):
            mask = result['predictions'][0].pred_instances.masks[i]
            mask = np.array(mask).astype(np.uint8) * 255
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

            ### score 값이 0.8이하이거나 label이 2개로 나누어 지는것이 아닌 경우는 continue
            if result['predictions'][0].pred_instances.scores[i] <= 0.8 or num_labels <= 2:
                continue

            # color = np.random.randint(0, 256, size=(3,), dtype=np.uint8)
            # color_img = np.tile(color, (h, w, 1))
            # color_img = cv2.bitwise_and(color_img, color_img, mask=mask)
            # mask_img = cv2.copyTo(color_img, mask, mask_img)
            
            # x1, y1, x2, y2 = result['predictions'][0].pred_instances.bboxes[i]
            # color = color.tolist()
            # mask_img = cv2.rectangle(mask_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            # cv2.imwrite(os.path.join('test_img', test_img),mask_img)
            # mask_img = cv2.resize(mask_img, (int(w/4), int(h/4)))
            # cv2.imshow("result", mask_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()


            #### 이미지 처리 test
            # label의 class 명
            label_class = classes[result['predictions'][0].pred_instances.labels[i]]

            # label별로 분할된 segmentation mask의 정보를 담을 리스트
            # [index_number, area, [center_x, center_y], [bbox]]
            label_info = []
            for j in range(num_labels):
                l_x, l_y, l_w, l_h, area = stats[j]

                if area <= 100:
                    continue
                
                # mask_img2 = cv2.rectangle(mask_img2, (l_x, l_y, l_w, l_h), color, 5)
                label_info.append([j, area, [centroids[j][0], centroids[j][1]], [l_x, l_y, l_w, l_h]])
        
            label_info.sort(key=lambda x:x[1], reverse=True)
            
            if len(label_info) <= 2:
                continue

            ### 가장 넓은 면적은 배경으로 처리하여 삭제
            label_info = label_info[1:]
            central_label = label_info[0]

            ### class별로 다른 알고리즘을적용, (fruit, flower, bud flower, growing, node, cap)은 가장 큰 mask 하나만을 남김
            if label_class in ['fruit', 'flower', 'bud_flower', 'growing', 'node', 'cap']:
                ### 가장 큰 mask의 index 번호 추출
                central_label_index = central_label[0]
                n_mask = np.where(labels == central_label_index, 1, 0)
                n_mask = np.array(n_mask).astype(np.uint8) * 255

                color = np.random.randint(0, 256, size=(3,), dtype=np.uint8)
                color_img = np.tile(color, (h, w, 1))
                color_img = cv2.bitwise_and(color_img, color_img, mask=n_mask)
                mask_img2 = cv2.copyTo(color_img, n_mask, mask_img2)
                # color = color.tolist()
                # mask_img2 = cv2.rectangle(mask_img2, (l_x, l_y, l_w, l_h), color, 5)

            ### leaf, midrib, stem은 끊어진 mask를 하나로 연결
            else:
                ### 가장 큰 mask의 index 번호 추출
                central_label_index = central_label[0]
                n_mask = labels.copy()
                print(f"mask : {n_mask.shape}")
                for label in label_info[1:]:
                    flag = check_bboxex_intersect(central_label[3], label[3])
                    if not flag:
                        n_mask = np.where(labels == label[0], 0, n_mask)
                        continue
                    else:
                        n_mask = np.where(labels == label[0], central_label_index, n_mask)
                print(f"n mask : {n_mask.shape}")

                n_mask = np.array(n_mask).astype(np.uint8) * 255
                contour = connect_contour(mask=n_mask)
                color = np.random.randint(0, 256, size=(3,), dtype=np.uint8)
                color_img = np.tile(color, (h, w, 1))
                color_img = cv2.bitwise_and(color_img, color_img, mask=n_mask)
                mask_img2 = cv2.copyTo(color_img, n_mask, mask_img2)
                mask_img2 = cv2.drawContours(mask_img2, contour, -1, (255,255,255), 5)
                resize_img = cv2.resize(mask_img2, (int(w/2), int(h/2)))
                cv2.imshow("zero_img", resize_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


        # mask_img2 = cv2.resize(mask_img2, (int(w/4), int(h/4)))
        # cv2.imshow("zero_img", mask_img2)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

def check_bboxex_intersect(bbox1, bbox2):
    '''
        두 개의 bbox가 겹치는 지 확인 (bbox : x1, y1, w, h)
    '''
    print(f"box1:{bbox1}, box2:{bbox2}")
    x1_overlap = max(bbox1[0], bbox2[0])
    y1_overlap = max(bbox1[1], bbox2[1])
    x2_overlap = max(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    y2_overlap = max(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
    
    # 겹치는 영역의 좌표 계산
    width = x2_overlap - x1_overlap
    height = y2_overlap - y1_overlap
    print(width >= 0 and height >= 0)
    return width >= 0 and height >= 0

def distance_between_mask(mask1, mask2):
    '''
        두 마스크 사이의 최소 거리 계산
    '''
    # 각 마스크의 컨투어 계산
    contour1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dist = cv2.contourArea(contour1[0], True)
    for cnt in contour2:
        dist = min(dist, cv2.pointPolygonTest(cnt, (cx1, cy1), True))
    return dist


def connect_contour(mask):
    ### mask의 가장 바깥 컨투어
    print(np.max(mask))
    indices = np.where(mask == np.max(mask))
    coordinates = list(zip(indices[1], indices[0]))
    img = cv2.fillPoly(np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8), [np.array(coordinates)], (255,255,255))
    # contour, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    resize_img = cv2.resize(img, (int(img.shape[0]/2), int(img.shape[0]/2)))
    cv2.imshow('poly', resize_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 

if __name__ == '__main__':
    main()