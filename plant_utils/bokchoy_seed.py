# plant_type = 'bokchoy_seed'
# ### load parts list corresponding to plant type.
# parts_list = json.load(open(
#     os.path.join(os.getcwd(), "for_inference", json.load(open("metadata.json", "r"))[plant_type]["parts_list"]),
#     "r"))
# .`

import cv2
import numpy as np

from detectron2.utils.visualizer import GenericMask
from plant_utils.common import *
from utils import (get_length, get_slope_alpha, create_check_flag, get_width_point)
import json

class Bokchoy_seed:
    def __init__(self, config, class_name_list):
        self.polygons = list()
        self.boxes = list()
        self.useful_mask_idx = []
        self.count_object = dict()
        self.bbox = dict()  # bbox를 저장할 dict
        self.center_point = dict()  # 중앙점을 저장할 dict
        self.idx_dict = dict()
        self.segmentations = dict()
        self.outer_inner_idx = dict()
        self.point_dict = dict()
        self.config = config
        self.plant_type = 'bokchoy_seed'
        self.class_name_list = class_name_list
        self.resize_scale = self.config.RESIZE_SCALE  # image가 너무 크면 resize_num 만큼 나눈다.
        self.margin_error = self.config.MARGIN_ERROR  # 오차범위
        self.centerbox_ratio = self.config.FRUIT_CENTER_BOX_RATIO  # fruit에서 cap이 fruit의 중앙의 얼마만큼의 범위 내에 속하는지를 결정하는 값
        self.img = None
        if self.plant_type in VALID_PLANTS:
            self.set_dict()

    def pop_none_object(self):
        pop_list = []
        for key, item in self.segmentations.items():
            if len(item) == 0:
                pop_list.append(key)

        for pop_ele in pop_list:
            self.segmentations.pop(pop_ele)

    def set_outer_inner_idx(self):
        # 외부 object와 내부 object사이의 상관관계를 계산하여 서로 대응되는 외부-내부 object의 index끼리 묶는다.
        # 예: 과일 5개와 꼭지 3개가 detection되었다면, 5개의 과일 중 3개는 꼭지를 포함한 과일일 것이다.
        # 이 때 3개의 과일과 3개의 꼭지의 대응 여부를 확인하여 해당 obejct의 index끼리 묶는다.
        # outer_inner_idx[object_name].keys(): [outer_idx_1, outer_idx_2, ..., outer_idx_n]
        # outer_inner_idx[object_name][outer_idx_1]: [inner_idx_1, inner_idx_2, ... inner_idx_n]
        for outer_objects in outer_objects_bokchoy_seed:  # 외부 object를 for문으로 나열
            self.outer_inner_idx[outer_objects] = dict()
            for outer_info in self.bbox[outer_objects]:  # 외부 object의 index를 나열
                bbox, outer_idx = outer_info
                self.outer_inner_idx[outer_objects][f'{outer_idx}'] = list()
                x_min, y_min, x_max, y_max = bbox
                width = (x_max - x_min) / 2
                height = (y_max - y_min) / 2

                # x_min, x_max, y_min, y_max : outer object의 bbox영역
                add_area_x_min = 0
                add_area_y_min = 0
                add_area_x_max = 0
                add_area_y_max = 0
                if outer_objects == "midrib":
                    # 포함 area를 각 width, height의 1/5만큼 씩 잘라낸다. (정확한 midrid를 가려내기 위해)
                    add_area_x_min = -width / 5
                    add_area_y_min = -height / 5
                    add_area_x_max = -width / 5
                    add_area_y_max = -height / 5
                elif outer_objects == "fruit":
                    # 포함 area를 height의 1/4만큼 올린다. (cap은 보통 위에 위치)
                    # image는 위가 y값의 시작이다
                    add_area_y_min = +height / 4
                    add_area_y_max = -height / 4

                for inner_objects in inner_objects_bokchoy_seed:  # 내부 object를 나열
                    for inner_info in self.center_point[inner_objects]:  # 각 내부 object의 중앙점을 나열
                        center_point, inner_idx = inner_info
                        x_center, y_center = center_point

                        # 내부 object의 중앙 점이 외부 object의 영역 안에 위치하는지 확인
                        # 위치하는 경우 외부-내부 object간 대응된다고 판단
                        if x_min - add_area_x_min < x_center < x_max + add_area_x_max and \
                                y_min - add_area_y_min < y_center < y_max + add_area_y_max:
                            if outer_idx not in self.useful_mask_idx:
                                self.useful_mask_idx.append(outer_idx)
                            self.useful_mask_idx.append(inner_idx)

                            # 서로 대응되는 object의 index를 key값 또는 list의 요소로 저장
                            self.outer_inner_idx[outer_objects][f'{outer_idx}'].append([inner_idx, inner_objects])
                                                # else:
                        #     self.outer_inner_idx[outer_objects][f'{outer_idx}'].append([None, None])

    def set_dict(self):

        # 가장자리 points를 저장할 dict
        for object_name in SEG_OBJECT_LIST:
            self.segmentations[object_name] = []

        # index를 저장할 dict
        for object_name in plant_object_idx_bokchoy_seed:
            self.idx_dict[object_name] = []

            # self.set_outer_inner_idx 에서 사용됨
        for object_name in inner_objects_bokchoy_seed:
            self.center_point[object_name] = []
        for object_name in outer_objects_bokchoy_seed:
            self.bbox[object_name] = []

        # object의 개수를 저장할 dict
        for obj_name in count_object_bokchoy_seed:
            self.count_object[obj_name] = 0

        # 좌표 계산에 사용할 mask의 idx만 list로 저장

    def find_coordinate_slicing(self, continue_check, x, y, _slope, _alpha, edge_list, continue_num, skip_num,
                                using_margin_error=True, margin_error=3):
        if using_margin_error:
            margin_error = self.margin_error

        if continue_check:
            # int(y) == int(_slope*x + _alpha) 만으로는 1차함수 값에 대응되는 x, y값이 없는 경우가 있다.
            # self.margin_error 를 사용해서 오차범위를 조절해가며 1차함수 값에 대응되는 x, y값을 찾아본다.
            if (int(_slope * x + _alpha) - margin_error <= int(y) <= int(_slope * x + _alpha) + margin_error) or \
                    (int(_slope * (x - margin_error) + _alpha) <= int(y) <= int(_slope * (x + margin_error) + _alpha)):
                edge_list.append([x, y])
                continue_check = False
        else:
            if continue_num == 0:
                continue_num = skip_num
                continue_check = True
            else:
                continue_num -= 1

        return edge_list, continue_num, continue_check

    def check_outer_exist(self, object_name):
        # if object_name not in list(self.outer_inner_idx.keys()):
        #     raise KeyError(f"{object_name} not in self.outer_inner_idx "
        #                    f"plant type is {self.plant_type}")

        if self.outer_inner_idx.get(object_name, None) is None:
            return False
        if len(list(self.outer_inner_idx[object_name].keys())) != 0:
            return True
        else:
            return False

    def get_object_idx(self, object_name):
        idx_list = []
        for idx in self.idx_dict[object_name]:
            if idx not in self.useful_mask_idx:
                continue
            idx_list.append(idx)

        return idx_list

    def contain_idx2dict(self, outputs):
        """
            self.idx_dict에 각각의 class name에 해당하는 key를 정의하고
            해당 class name에 부합하는 idx를 key값에 list로 할당
        """
        # leaf에 포함되는 midrid를 찾기 위해 leaf와 midrid를 구분
        for idx, output in enumerate(outputs["instances"].pred_classes):

            for object_name in plant_object_idx_bokchoy_seed:
                if self.class_name_list[output] == object_name:
                    print(f'Detected part: {idx} - {object_name}')
                    self.idx_dict[object_name].append(idx)  # 해당 idx를 저장

                    if object_name in inner_objects_bokchoy_seed:
                        # inner object는 서로 대응관계가 있는 object인지 확인하기 전 까진 self.useful_mask_idx에 append하지 않는다.
                        self.center_point[object_name].append([compute_center_point(self.boxes[idx]), idx])
                    elif object_name in outer_objects_bokchoy_seed:
                        self.bbox[object_name].append([self.boxes[idx], idx])
                        if object_name == "fruit":  # fruit는 대응관계인 cap이 없어도 좌표찾기를 적용할 것이기 때문에 append
                            self.useful_mask_idx.append(idx)
                            # leaf인 경우는 midrid와 대응되지 않는 경우 좌표찾기 적용하지 않을 예정
                    else:
                        self.useful_mask_idx.append(idx)

                    if object_name in count_object_bokchoy_seed:
                        self.count_object[object_name] += 1

    def set_boxes_polygons(self, height, width, predictions):
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None

        for box in boxes.tensor.numpy():
            self.boxes.append([int(box[0]), int(box[1]), int(box[2]), int(box[3])])

        masks = np.asarray(predictions.pred_masks)
        masks = [GenericMask(x, height, width) for x in masks]

        # 모든 polygon(가장자리 points)를 list에 저장
        for mask in masks:
            points = list()
            for segment in mask.polygons:
                # segment : coordinates of segmentation boundary
                x_coordinates = segment.reshape(-1, 2)[:, 0]  # segmentation에서 x좌표만 extract
                y_coordinates = segment.reshape(-1, 2)[:, 1]  # segmentation에서 y좌표만 extract

                x_coordinates, y_coordinates = (list(map(int, x_coordinates / self.resize_scale)),
                                                list(map(int,
                                                         y_coordinates / self.resize_scale)))  # adjust coordinates for resized image

                for x, y in zip(x_coordinates, y_coordinates):
                    points.append([x, y])

            self.polygons.append(points)

    def set_segmentations_dict(self, idx):
        for object_name in result_objects_bokchoy_seed:
            if object_name not in SEG_OBJECT_LIST:
                continue
            if idx in self.idx_dict[object_name]:
                if object_name in count_object_bokchoy_seed:
                    x_min, y_min, x_max, y_max = self.boxes[idx]
                    self.segmentations[object_name].append([int(x_min), int(y_min), int(x_max), int(y_max)])
                else:
                    self.segmentations[object_name].append(self.polygons[idx].copy())

    # main function
    def calculate_coordinates(self, outputs, img):
        if self.plant_type not in VALID_PLANTS:
            return None, None, None, None
        predictions = outputs["instances"].to("cpu")
        if not predictions.has("pred_masks"):
            # mask가 한 개도 없으면 save file을 하지 않음
            has_mask = False
            return img, has_mask, None, None
        height = img.shape[0]
        width = img.shape[1]
        self.img = cv2.resize(img, (int(width / self.resize_scale), int(height / self.resize_scale)))

        masks = np.asarray(predictions.pred_masks)
        boxes = np.asarray(predictions.pred_boxes)
        for idx, mask in enumerate(masks):
            mask = np.array(mask).astype(np.uint8) * 255
            n_mask, n_bbox = self.remove_seperate_mask(mask, predictions.pred_classes[idx])
            masks[idx] = n_mask
            boxes[idx] = n_bbox

        predictions.pred_masks = masks
        predictions.pred_bbox = boxes
        
        # 모든 polygon(가장자리 points)를 list에 저장
        self.set_boxes_polygons(height, width, predictions)  # make self.boxes, self.polygons
        self.contain_idx2dict(outputs)  # 모든 object의 index를 dict에 저장
        self.set_outer_inner_idx()  # 서로 대응되는 object의 index를 dict의 key값과 그 안의 list의 요소로 저장

        for idx, _ in enumerate(self.boxes):
            if len(self.polygons[idx]) == 0:
                continue  # object가 detecting은 됐지만 points(polygon)가 없는 경우도 있다.
            self.set_segmentations_dict(idx)  # self.segmentations에 각 object name별로 index저장

        # 여기서부터 좌표 계산
        coordinates_dict = {}  # 각 계산한 좌표를 저장할 dict
        # if len(self.get_object_idx("length")) != 0:
        #     coordinates_dict["stem"] = self.get_stem_seed_info()
        if self.check_outer_exist("leaf"):
            coordinates_dict["leaf"] = self.get_draw_leaf_info()

        if self.check_outer_exist("fruit"):
            fruit_point_dict = self.get_draw_fruit_info()
            coordinates_dict["fruit"] = fruit_point_dict

        # object counting
        for object_name in count_object_bokchoy_seed:
            bbox_list = list()
            for object_idx in self.idx_dict[object_name]:
                bbox_list.append(self.boxes[object_idx])

            if self.count_object[object_name] == 0:
                continue
            coordinates_dict[object_name] = dict(count=self.count_object[object_name],
                                                 # 1개의 image에 detecting된 object_name의 개수
                                                 bbox=bbox_list)

        has_mask = True

        self.pop_none_object()
        return img, has_mask, coordinates_dict, self.segmentations

    def get_stem_seed_info(self):
        tmp_list = []

        for idx in self.get_object_idx("length"):
            stem_coordinates = self.polygons[idx].copy()
            midrid_center_coordinates = get_sorted_center_points(self.polygons[idx].copy(),
                                                                      width_or_height=return_width_or_height(
                                                                          self.boxes[idx]))
            tmp_dict = {}
            tmp_dict["segmentation"] = stem_coordinates.copy()
            tmp_dict["type"] = self.plant_type
            tmp_dict["bbox"] = self.boxes[idx]
            # tmp_dict["width"]

            midrid_center_coordinates_part_1 = midrid_center_coordinates[:len(midrid_center_coordinates) // 2]
            midrid_center_coordinates_part_2 = midrid_center_coordinates[len(midrid_center_coordinates) // 2:]

            point_list = []
            num_spot = self.config.NUM_SPOT  # c_cfg.NUM_SPOT == 10 이면 10개의 point를 찍는다.
            num_spot_1 = int(num_spot / 3)

            count_bot = 1
            ### select particular point
            for i, center_coordinate in enumerate(midrid_center_coordinates_part_1):
                if i == int(len(midrid_center_coordinates_part_1) * (count_bot / num_spot_1)) or i == 0 or i == (
                        len(midrid_center_coordinates_part_1) - 1):
                    count_bot += 1
                    point_list.append(center_coordinate)

            num_spot_2 = int(num_spot / 5)

            count_bot = 1
            ### select particular point
            for i, center_coordinate in enumerate(midrid_center_coordinates_part_2):
                if i == int(len(midrid_center_coordinates_part_2) * (count_bot / num_spot_2)) or i == 0 or i == (
                        len(midrid_center_coordinates_part_2) - 1):
                    count_bot += 1
                    point_list.append(center_coordinate)

            tmp_dict["height"] = point_list

            midpoint_of_last_y_point = (point_list[-1][1] + point_list[-2][1]) // 2

            stem_coordinates.sort(key=lambda x: x[1])
            last_part_stem_of_lastpoint = stem_coordinates[int(len(stem_coordinates) * 4 / 5):]

            for i in range(5):  # 오차
                min_x_point, max_x_point = 100000, -1

                for point in last_part_stem_of_lastpoint:
                    # print(f"midpoint_of_lastpoint : {midpoint_of_lastpoint},        point : {point}")
                    if midpoint_of_last_y_point == point[1]:  # midpoint_of_last_y_point와 같은 y값을 가진 좌표인 경우
                        if min_x_point > point[0]:  # 해당 x좌표가 줄기의 왼쪽 부분일 경우
                            min_x_point = point[0]
                        if max_x_point < point[0]:  # 해당 x좌표가 줄기의 오른쪽 부분일 경우
                            max_x_point = point[0]

                if min_x_point == max_x_point or max_x_point == -1:  # midpoint_of_last_y_point와 같은 y값을 가진 좌표가 두 개 미만으로 탐색된 경우
                    midpoint_of_last_y_point += 1  # midpoint_of_last_y_point의 y좌표 증가
                else:
                    tmp_dict["width"] = [[min_x_point, midpoint_of_last_y_point],
                                         [max_x_point, midpoint_of_last_y_point]]
                    break

            if "width" not in tmp_dict.keys() or "height" not in tmp_dict.keys(): continue

            tmp_list.append(tmp_dict)

        return tmp_list

    def get_stem_points(self, stem_points, stem_info, w_h):
        if w_h == "height":
            stem_points.sort(key=lambda _x: _x[1])
        else:
            stem_points.sort()

        stem_coordinates_sorted = stem_points
        # stem의 중단부분 coordinates       1/4 ~ 3/4
        stem_coordinates_mid = stem_coordinates_sorted[
                               int(len(stem_coordinates_sorted) / 4):int(len(stem_coordinates_sorted) / 4) * 3]
        # stem의 중단 영역의 top부분(또는 좌측 부분) coordinates    1/4 ~ 1.3/4
        stem_coordinates_mid_top = stem_coordinates_sorted[
                                   int(len(stem_coordinates_sorted) / 4):int((len(stem_coordinates_sorted) / 4) * 1.3)]
        # stem의 중단 영역의 bottom부분(또는 우측 부분) coordinates     2.7/4 ~ 3/4
        stem_coordinates_mid_bottom = stem_coordinates_sorted[int((len(stem_coordinates_sorted) / 4) * 2.7):int(
            len(stem_coordinates_sorted) / 4) * 3]
        # stem의 중단부분 coordinates(넓은 영역)    1/5 ~ 4/5
        stem_coordinates_mid_wide = stem_coordinates_sorted[
                                    int(len(stem_coordinates_sorted) / 5):int(len(stem_coordinates_sorted) / 5) * 4]

        x_min, y_min, x_max, y_max = stem_info["bbox"]
        x_center, y_center = [int((x_min + x_max) / 2), int((y_min + y_max) / 2)]
        box_width, box_height = x_max - x_min, y_max - y_min

        x_bottom_left = None
        x_top_right = None
        y_left_bottom = None
        y_right_top = None
        if w_h == "height":  # stem이 수직 형태인 경우
            # stem이 우측으로 기울어졌는지, 좌측으로 기울어졌는지 확인
            x_top_left = x_max
            y_top = None
            x_bottom_right = x_min
            y_bottom = None
            for x_stem, y_stem in stem_coordinates_mid_top:
                if x_top_left > x_stem:
                    x_top_left = x_stem
                    y_top = y_stem
                    # [x_top_left, y_top] : stem의 1/4지점 좌측 상단 point
            for x_stem, y_stem in stem_coordinates_mid_bottom:
                if x_bottom_right < x_stem:
                    x_bottom_right = x_stem
                    y_bottom = y_stem
                    # [x_bottom_right, y_bottom] : stem의 3/4지점 우측 하단 point

            count = 0
            for x_stem, y_stem in stem_coordinates_mid_wide:
                if count == 2:
                    break
                if y_stem == y_bottom and x_stem != x_bottom_right:
                    x_bottom_left = x_stem  # [x_bottom_left, y_bottom] : stem의 3/4지점 좌측 하단 point
                    count += 1
                    continue
                if y_stem == y_top and x_stem != x_top_left:
                    x_top_right = x_stem  # [x_top_right, y_top] : stem의 1/4지점 우측 상단 point
                    count += 1
                    continue

        else:  # stem이 수평 형태인 경우
            # stem의 우측이 아래방향인지, 좌측이 아래방향인지 확인
            y_left_top = y_min
            x_top = None
            y_right_bottom = y_max
            x_bottom = None
            for x_stem, y_stem in stem_coordinates_mid_top:
                if y_left_top < y_stem:
                    y_left_top = y_stem
                    x_top = x_stem
                    # [x_left, y_left_top] : stem의 1/4지점 좌측 상단 point
            for x_stem, y_stem in stem_coordinates_mid_bottom:
                if y_right_bottom < y_stem:
                    y_right_bottom = y_stem
                    x_bottom = x_stem
                    # [x_bottom, y_right_bottom] : stem의 3/4지점 우측 하단 point

            y_left_bottom = None
            y_right_top = None
            count = 0
            for x_stem, y_stem in stem_coordinates_mid_wide:
                if count == 2:
                    break
                if x_stem == x_bottom and y_stem != y_right_bottom:
                    y_left_bottom = y_stem  # [x_bottom, y_left_bottom] : stem의 3/4지점 좌측 하단 point
                    count += 1
                    continue
                if x_stem == x_top and y_stem != y_left_top:
                    y_right_top = y_stem  # [x_top, y_left_top] : stem의 1/4지점 우측 상단 point
                    count += 1
                    continue

        if (x_bottom_left is not None and x_top_right is not None) or (
                y_right_top is not None and y_left_bottom is not None):

            if w_h == "height":
                # slope_left : stem의 좌측 point의 기울기
                # slope_right : stem의 우측 point의 기울기
                if x_top_right - x_bottom_right == 0 or x_top_left - x_bottom_left == 0:  # 완전한 수직인 경우
                    slope_left = 100
                    slope_right = 100
                else:
                    slope_left = (y_top - y_bottom) / (x_top_left - x_bottom_left)
                    slope_right = (y_top - y_bottom) / (x_top_right - x_bottom_right)

                if slope_right * slope_left > 0:
                    slope = (slope_left + slope_right) / 2  # 두 기울기가 같은 방향으로 기울어져 있을 경우
                else:
                    slope = 100  # 두 기울기가 서로 음-양 값을 가진 경우
            elif w_h == "width":
                if y_right_bottom - y_left_bottom == 0 or y_left_top - y_right_top == 0:  # 완전한 수평인 경우
                    slope_top = 0
                    slope_bottom = 0
                else:
                    slope_top = (x_top - x_bottom) / (y_left_top - y_right_top)
                    slope_bottom = (x_top - x_bottom) / (y_right_bottom - y_left_bottom)

                if slope_bottom * slope_top > 0:
                    slope = (slope_bottom + slope_top) / 2  # stem이 완전한 수평이 아닌, 기울어진 수평일 경우
                else:
                    slope = 0  # stem이 완전한 수평인 경우

            if abs(slope) <= 0.1:
                inverse_slope = 10  # stem이 거의 수평인 경우
            elif abs(slope) >= 10:
                inverse_slope = 0.1  # stem이 거의 수직인 경우
            else:
                inverse_slope = -1 / slope
                alpha = -1 * x_center * slope + y_center
                inverse_alpha = -1 * x_center * inverse_slope + y_center

            edge_list = []
            _width_coordinates = None
            for x, y in stem_coordinates_mid:
                if abs(inverse_slope) >= 10:
                    if x_center == x:
                        edge_list.append([x, y])
                elif abs(inverse_slope) <= 0.1:
                    if y_center == y:
                        edge_list.append([x, y])
                else:
                    if int(inverse_slope * x + inverse_alpha) - 1 <= int(y) <= int(
                            inverse_slope * x + inverse_alpha) + 1:
                        edge_list.append([x, y])

                if len(edge_list) == 2:
                    x_1, y_1, x_2, y_2 = edge_list[0][0], edge_list[0][1], edge_list[1][0], edge_list[1][1]

                    # if (w_h == "h" and abs(x_1 - x_2) < box_width/5) or (w_h == "w" and abs(y_1 - y_2) < box_height/5):
                    #     edge_list.pop(-1)
                    #     continue
                    if get_length([x_1, y_1], [x_2, y_2]) < 5:
                        edge_list.pop(-1)
                        continue
                    else:
                        _width_coordinates = edge_list

            if _width_coordinates is not None:
                stem_info["width"] = _width_coordinates

                width_1, width_2 = _width_coordinates[0], _width_coordinates[1]
                x_center_point, y_center_point = (width_1[0] + width_2[0]) // 2, (width_1[1] + width_2[1]) // 2
                stem_info["center"] = [x_center_point, y_center_point]

                stem_coordinates_top_bottom = stem_coordinates_sorted[
                                              :int(len(stem_coordinates_sorted) / 5)]  # stem의 상단 또는 좌측
                for coordinate in stem_coordinates_sorted[
                                  int((len(stem_coordinates_sorted) / 5) * 4):]:  # stem의 하단 또는 우측
                    stem_coordinates_top_bottom.append(
                        coordinate)  # stem_coordinates_top_bottom : stem의 양 끝 부분들만 분리

                for i in range(self.margin_error):
                    continue_check, continue_num, edge_list, skip_num = create_check_flag(
                        stem_coordinates_top_bottom)
                    for x, y in stem_coordinates_top_bottom:
                        if abs(slope) <= 0.1:
                            if y_center == y: edge_list.append([x, y])
                        elif abs(slope) >= 10:
                            if x_center == x: edge_list.append([x, y])
                        else:
                            edge_list, continue_num, continue_check = self.find_coordinate_slicing(continue_check,
                                                                                                   x, y, slope,
                                                                                                   alpha, edge_list,
                                                                                                   continue_num,
                                                                                                   skip_num, False,
                                                                                                   i)

                        if len(edge_list) == 2:
                            x_1, y_1, x_2, y_2 = edge_list[0][0], edge_list[0][1], edge_list[1][0], edge_list[1][1]
                            length = get_length([x_1, y_1], [x_2, y_2])

                            if (w_h == "h" and length < box_height / 4) or (
                                    w_h == "w" and length < box_width / 4):
                                edge_list.pop(-1)
                                continue
                            else:
                                stem_info["height"] = edge_list
                                break
                        else:
                            continue
                    if "height" not in stem_info.keys():
                        continue
                    else:
                        # 2개의 point였던 height에 center point를 추가
                        tmp_list = stem_info["height"]
                        stem_info["height"] = [tmp_list[0], stem_info["center"], tmp_list[1]]
                        break
        return stem_info

    def get_draw_fruit_info(self):
        """
        cap_fruit_meta_list : cap과 fruit의 idx와 boundary coordinates를 담은 list
                            : [[idx_cap_1, coordinates_1], [idx_fruit_2, coordinates_2], ...]
                            cap과 fruit의 순서는 불규칙
        """

        # Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated.
        # If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
        # NumPy 버전이 1.19 이상에서 API가 deprecated 되어서 발생하는 warning
        # 당장 사용하는데는 문제가 없음.
        # 만약 warning이 보기 싫으시면 NumPy 1.18.5 버전으로 다운그레이

        fruit_point_dict = dict(fruit_only=[],
                                cap_fruit_side=[],
                                cap_fruit_above=[])

        for outer_idx, inner_info_list in self.outer_inner_idx['fruit'].items():
            outer_idx = int(outer_idx)
            # print(f"outer_idx : {outer_idx}")
            # print(f"fruit_points : {self.polygons[outer_idx]}")
            # img = self.img
            # for point in self.polygons[outer_idx]:
            #     cv2.circle(img, point, radius=2, color=(0, 255, 255), thickness=-1)

            # cv2.imshow("img", img)
            # while True:
            #     if cv2.waitKey() == 27: break

            if len(inner_info_list) == 0:  # cap(꼭지)가 포함되지 않는 경우
                fruit_only_dict = self.get_only_fruit_points(outer_idx)
                if fruit_only_dict is not None:
                    fruit_point_dict["fruit_only"].append(fruit_only_dict)
            elif len(inner_info_list) >= 1:  # cap또는 cap_2가 1개 이상 포함된 경우
                fruit_point_dict = self.get_cap_fruit_points(outer_idx, inner_info_list, fruit_point_dict)
        return fruit_point_dict

    def get_only_fruit_points(self, idx):
        fruit_points = self.polygons[idx].copy()
        fruit_points_for_search = self.polygons[idx].copy()
        bbox_fruit = self.boxes[idx]

        fruit_only_info = dict(segmentation=fruit_points.copy(),
                               type=self.plant_type,
                               bbox=bbox_fruit)

        fruit_only_info = self.common_fruit_only(fruit_points, fruit_points_for_search, fruit_only_info)

        return fruit_only_info

    def common_fruit_only(self,
                          fruit_points,
                          fruit_points_for_search,
                          fruit_only_info):

        # sort according to x coordinatess
        fruit_points.sort()

        fruit_sorted_coordinates_top = []
        fruit_sorted_coordinates_bottom = []
        # x_coordinate_slicing 에는 가장 작은 x부터 가장 큰 x까지 slicing하며,
        # 각 x에 대해서 대응되는 두 개의 y의 위쪽 좌표(y_coordinate_max), 아래쪽 좌표(y_coordinate_min)를 얻는다.
        x_coordinate_slicing, y_coordinate_max, y_coordinate_min = None, -1, 1000000
        for x_y_coordinate in fruit_points:  # fruit_coordinates 를 x값이 낮은 위치부터 slicing
            if x_coordinate_slicing != x_y_coordinate[0]:
                fruit_sorted_coordinates_top.append([x_coordinate_slicing, y_coordinate_max])
                fruit_sorted_coordinates_bottom.append([x_coordinate_slicing, y_coordinate_min])
                x_coordinate_slicing, y_coordinate_max, y_coordinate_min = x_y_coordinate[0], x_y_coordinate[1], \
                                                                           x_y_coordinate[1]
            elif x_coordinate_slicing is None:  # slicing을 처음 시작할 때
                fruit_sorted_coordinates_top.append([x_coordinate_slicing, y_coordinate_max])
                fruit_sorted_coordinates_bottom.append([x_coordinate_slicing, y_coordinate_min])
                x_coordinate_slicing, y_coordinate_max, y_coordinate_min = x_y_coordinate[0], x_y_coordinate[1], \
                                                                           x_y_coordinate[1]
            else:  # 같은 x좌표가 여러개인 경우, 그 중 y가 가장 큰 좌표와 작은 좌표를 얻어낸다. (x좌표 중복 제거)
                if y_coordinate_max < x_y_coordinate[1]:
                    y_coordinate_max = x_y_coordinate[1]
                if y_coordinate_min > x_y_coordinate[1]:
                    y_coordinate_min = x_y_coordinate[1]

        # fruit_sorted_coordinates_top : fruit의 위쪽 coordinates
        # fruit_sorted_coordinates_bottom : fruit의 아래쪽 coordinates
        check_idx = 0
        length_fruit = -1
        tmp_idx = 0
        if len(fruit_sorted_coordinates_top) == len(fruit_sorted_coordinates_bottom):
            # 두 list의 len이 같다면 바로 length계산
            for top_coordi, bottom_coordi in zip(fruit_sorted_coordinates_top, fruit_sorted_coordinates_bottom):
                tmp_idx += 1
                # 가장 큰 length를 찾아 index를 보관 후 좌표를 얻는다.
                if length_fruit < top_coordi[1] - bottom_coordi[1]:
                    length_fruit = top_coordi[1] - bottom_coordi[1]
                    check_idx = tmp_idx
        else:
            # 두 list의 len이 같지 않다면 작은 len을 가진 list를 기준으로
            min_len = min(len(fruit_sorted_coordinates_top), len(fruit_sorted_coordinates_bottom))
            add_value_bottom = 0
            add_value_top = 0
            for i in range(min_len):
                if fruit_sorted_coordinates_top[i + add_value_top][0] == \
                        fruit_sorted_coordinates_bottom[i + add_value_bottom][0]:
                    tmp_idx += 1
                    if length_fruit < fruit_sorted_coordinates_top[i + add_value_top][1] - \
                            fruit_sorted_coordinates_bottom[i + add_value_bottom][1]:
                        length_fruit = fruit_sorted_coordinates_top[i + add_value_top][1] - \
                                       fruit_sorted_coordinates_bottom[i + add_value_bottom][1]
                        check_idx = tmp_idx
                    # 높은 x값을 가진 coordinates list는 다음 iteration에서 add_value에 의한 -를 통해 index를 유지시켜준다.
                elif fruit_sorted_coordinates_top[i + add_value_top][0] < \
                        fruit_sorted_coordinates_bottom[i + add_value_bottom][0]:
                    add_value_bottom -= 1
                elif fruit_sorted_coordinates_top[i + add_value_top][0] > \
                        fruit_sorted_coordinates_bottom[i + add_value_bottom][0]:
                    add_value_top -= 1

        x_1, y_1 = fruit_sorted_coordinates_top[check_idx]
        x_2, y_2 = fruit_sorted_coordinates_bottom[check_idx]

        fruit_only_info["height"] = [[x_1, y_1], [x_2, y_2]]

        ## find width point
        _slope, _ = get_slope_alpha(x_1, y_1, x_2, y_2)

        if _slope == 0:
            inverse_slope = 20
        else:
            inverse_slope = (-1) / _slope

        mid_point_alpha = []
        mid_point_num = 5
        x_val, y_val = abs(x_1 - x_2) / mid_point_num, abs(y_1 - y_2) / mid_point_num
        num_list = [i - mid_point_num // 2 for i in range(mid_point_num)]
        for i in num_list:
            x_mid, y_mid = (x_1 + x_2) / 2 + (x_val * i), (y_1 + y_2) / 2 + (y_val * i)
            _alpha = y_mid - x_mid * inverse_slope
            mid_point_alpha.append([_alpha, [int(x_mid), int(y_mid)]])

        width_length_list = []
        for mid_point in mid_point_alpha:
            _alpha = mid_point[0]

            check_boolean = False
            for i in range(self.margin_error):
                continue_check, continue_num, edge_list, skip_num = create_check_flag(fruit_points_for_search)
                for x, y in fruit_points_for_search:
                    edge_list, continue_num, continue_check = self.find_coordinate_slicing(continue_check, x, y,
                                                                                           inverse_slope, _alpha,
                                                                                           edge_list, continue_num,
                                                                                           skip_num, False, i)
                    if len(edge_list) == 2:
                        width_coordinates = [edge_list[0], edge_list[1]]
                        check_boolean = True
                        break
                if not check_boolean:
                    continue
                else:
                    break

            if width_coordinates is None:
                continue
            length = get_length(width_coordinates[0], width_coordinates[1])
            width_length_list.append([length, width_coordinates, mid_point[1]])

        # 폭(width) 좌표를 구하지 못하는 경우 return None
        if len(width_length_list) == 0:
            return None

        max_length = -1
        for width_info in width_length_list:
            width_length, width_coordi, mid_center_point = width_info
            if max_length < width_length:
                max_length = width_length

                fruit_only_info["width"] = width_coordi
                fruit_only_info["center"] = mid_center_point

        if "width" not in fruit_only_info.keys() or \
                "height" not in fruit_only_info.keys():
            return None

        return fruit_only_info

    def get_draw_leaf_info(self):
        """
        leaf의 boundary points와 midrid points, type, width points 등을 계산

        Return
            point_dict_list : [point_dict, point_dict, point_dict...]
                point_dict.keys() = ["midrid_point_coordinate", "leaf_width_edge_coordinates"]
                point_dict["midrid_point_coordinate"] : [point_1, point_2, ... point_n]
                point_dict["leaf_width_edge_coordinates"] : [edge_right, edge_left]   or  [edge_left, edge_right]
        """

        point_dict_list = []  # 각 leaf에 대해서 midrid의 points의 좌표, leaf의 폭 꼭지점의 좌표를 담게 된다.
        # [point_dict, point_dict, point_dict...]
        for outer_idx, inner_info_list in self.outer_inner_idx['leaf'].items():
            outer_idx = int(outer_idx)
            leaf_points = self.polygons[outer_idx].copy()
            bbox_leaf = self.boxes[outer_idx]
            self.point_dict = dict(midrid_point_coordinate=[],  # [point_1, point_2, ....]
                                   leaf_width_edge_coordinates=[],
                                   # [edge_right, edge_left]   or  [edge_left, edge_right]
                                   segmentation=leaf_points.copy(),
                                   type=self.plant_type,
                                   bbox=bbox_leaf)
            if len(inner_info_list) <= 0:
                height, width = self.get_leaf_width_height(bbox_leaf, leaf_points)
                self.point_dict["midrid_point_coordinate"] = height
                self.point_dict["leaf_width_edge_coordinates"] = width
                point_dict_list.append(self.point_dict)

            else:
                inner_idx, _ = inner_info_list[0]  # midrid는 각 leaf당 1개만 할당되어있다고 가정

                midrid_points = self.polygons[inner_idx].copy()
                bbox_midrid = self.boxes[inner_idx]
                more_longer = return_width_or_height(bbox_midrid)
                midrid_center_points = get_sorted_center_points(midrid_points,width_or_height=more_longer)

                if len(midrid_center_points) == 0:
                    height, width = self.get_leaf_width_height(bbox_leaf, leaf_points)
                    self.point_dict["midrid_point_coordinate"] = height
                    self.point_dict["leaf_width_edge_coordinates"] = width
                    point_dict_list.append(self.point_dict)
                    continue

                if more_longer:
                    midrid_center_points.sort(key=lambda x: x[1])
                else:
                    midrid_center_points.sort()


                # midrid의 point개수 설정
                num_spot = self.config.NUM_SPOT  # c_cfg.NUM_SPOT == 10 이면 10개의 point를 찍는다.
                num_spot = int(num_spot / 3)
                count_bot = 1
                ### select particular point
                # midrid point 확보
                for i, center_coordinate in enumerate(midrid_center_points):
                    if i == 0 or \
                            i == int(len(midrid_center_points) * (count_bot / num_spot)) or \
                            i == (len(midrid_center_points) - 1):
                        count_bot += 1
                        self.point_dict["midrid_point_coordinate"].append(center_coordinate)

                self.find_first_point_midrid(leaf_points, more_longer)
                self.find_last_point_midrid(leaf_points, more_longer)

                # calculate edge point about width of leaf base on point_coordinate
                check_availability, cross_point = self.find_width_point_midrid(leaf_points, bbox_leaf, more_longer)

                if len(self.point_dict["leaf_width_edge_coordinates"]) == 0:
                    point_dict_list.append(self.point_dict)
                    continue
                self.point_dict["leaf_width_edge_coordinates"] = get_width_point(
                    self.point_dict["leaf_width_edge_coordinates"], 5)

                if cross_point is not None:
                    self.point_dict["center"] = cross_point

                # midrid의 coordinates가 작은 영역에 뭉쳐있다면 point_dict_list에 append하지않음
                first_point, last_point = self.point_dict["midrid_point_coordinate"][0], \
                                          self.point_dict["midrid_point_coordinate"][-1]
                length = math.sqrt(
                    math.pow(first_point[0] - last_point[0], 2) + math.pow(first_point[1] - last_point[1], 2))

                box_width, box_height = compute_width_height(bbox_midrid)
                if more_longer == "width":
                    if length < box_width / 7:
                        check_availability = False
                elif more_longer == "height":
                    if length < box_height / 7:
                        check_availability = False

                if check_availability:
                    point_dict_list.append(self.point_dict)
        return point_dict_list

    def find_width_point_midrid(self, leaf_coordinates, bbox_leaf, width_or_height):
        # calculate center box coordinates
        x_min, y_min, x_max, y_max = bbox_leaf
        width, height = x_max - x_min, y_max - y_min

        if width_or_height == "height":
            x_min_center_box, x_max_center_box, y_min_center_box, y_max_center_box = x_min, x_max, y_min, int(y_max - height/3)
        else:
            x_min_center_box, x_max_center_box, y_min_center_box, y_max_center_box = x_min, x_max, y_min, y_max


        # -----TODO : for 박람회        # 2022-12-12
        # x_first, y_first = self.point_dict["midrid_point_coordinate"][0][0], self.point_dict["midrid_point_coordinate"][0][1]
        # x_last, y_last = self.point_dict["midrid_point_coordinate"][-1][0], self.point_dict["midrid_point_coordinate"][-1][1]
        # midrid_slope, _ = get_slope_alpha(x_first, y_first, x_last, y_last)        # midrid의 첫 번째 point와 마지막 point사이의 기울기

        # midrid_point를 한 번 솎아낸다.
        tmp_point_dict = []
        for idx in range(len(self.point_dict["midrid_point_coordinate"])):
            if idx == 0 or idx == len(self.point_dict["midrid_point_coordinate"]) - 1:
                tmp_point_dict.append(self.point_dict["midrid_point_coordinate"][idx])
                continue

            x_before, y_before = self.point_dict["midrid_point_coordinate"][idx - 1][0], \
                                 self.point_dict["midrid_point_coordinate"][idx - 1][1]
            x_this, y_this = self.point_dict["midrid_point_coordinate"][idx][0], \
                             self.point_dict["midrid_point_coordinate"][idx][1]
            x_next, y_next = self.point_dict["midrid_point_coordinate"][idx + 1][0], \
                             self.point_dict["midrid_point_coordinate"][idx + 1][1]

            slope_before, _ = get_slope_alpha(x_before, y_before, x_this, y_this)  # midrid의 현재 point와 이전 point사이의 기울기
            slope_next, _ = get_slope_alpha(x_this, y_this, x_next, y_next)  # midrid의 현재 point와 다음 point사이의 기울기

            if slope_before * slope_next < 0:  # 두 기울기의 곱이 음수인 경우 현재 point의 위치를 재조정
                x_npoint, y_npoint = int((x_before + x_next) / 2), int((y_before + y_next) / 2)
                tmp_point_dict.append([x_npoint, y_npoint])
            elif (abs(slope_before) < 1 and abs(slope_next) > 1) or (
                    abs(slope_before) > 1 and abs(slope_next) < 1):  # 기울기가 급격하게 변하는 경우 현재 point의 위치를 재조정
                x_npoint, y_npoint = int((x_before + x_next) / 2), int((y_before + y_next) / 2)
                tmp_point_dict.append([x_npoint, y_npoint])
            else:
                tmp_point_dict.append(self.point_dict["midrid_point_coordinate"][idx])

        self.point_dict["midrid_point_coordinate"] = tmp_point_dict


        edge_point_list = []
        for idx in range(len(self.point_dict["midrid_point_coordinate"])):
            if idx == 0 or idx == len(self.point_dict["midrid_point_coordinate"]) - 1: continue

            x_this, y_this = self.point_dict["midrid_point_coordinate"][idx][0], \
                             self.point_dict["midrid_point_coordinate"][idx][1]
            x_next, y_next = self.point_dict["midrid_point_coordinate"][idx + 1][0], \
                             self.point_dict["midrid_point_coordinate"][idx + 1][1]

            if not (
                    x_min_center_box < x_this < x_max_center_box and y_min_center_box < y_this < y_max_center_box): continue  # 현재 점이 center box안에 위치하지 않는 경우
            _slope, _ = get_slope_alpha(x_this, y_this, x_next, y_next)
            if _slope == 0:
                continue  # 기울기가 0인경우는 잘못 계산된 것
            inverse_slope = (-1) / _slope

            x_mid, y_mid = (x_this + x_next) // 2, (y_this + y_next) // 2  # midrid위의 각 point 사이의 중점
            coordinate_two_point = [[x_this, y_this], [x_mid, y_mid]]  # midrid위의 현재 point와, 중점 point


            for point in coordinate_two_point:
                x_coordinate, y_coordinate = point
                _alpha = y_coordinate - x_coordinate * inverse_slope  # 위에서 계산한 중점에서의 1차 함수의 절편

                # y = width_slope*x + _alpha : midrid위의 각 point로 표현되는 1차함수에 대해 기울기가 90도 차이나는 1차함수
                check_boolean = False
                for i in range(self.margin_error):
                    continue_check, continue_num, spot_list, skip_num = create_check_flag(leaf_coordinates)
                    for x, y in leaf_coordinates:
                        # 2022-12-12
                        if abs(inverse_slope) < 1 / 20:
                            if y_coordinate == y: spot_list.append([x, y])
                        elif abs(inverse_slope) > 20:
                            if x_coordinate == x: spot_list.append([x, y])
                        else:
                            spot_list, continue_num, continue_check = self.find_coordinate_slicing(continue_check,
                                                                                                   x, y,
                                                                                                   inverse_slope,
                                                                                                   _alpha, spot_list,
                                                                                                       continue_num,
                                                                                                       skip_num,
                                                                                                       False, i)
                        if len(spot_list) == 2:
                            length = math.sqrt(math.pow(spot_list[0][0] - spot_list[1][0], 2) + math.pow(spot_list[0][1] - spot_list[1][1], 2))
                            if length < 20:
                                spot_list.pop(-1)
                                continue

                            # spot_list[0] : width의 첫 번째 edge point
                            # spot_list[1] : width의 두 번째 edge point
                            # point : width의 두 edge point 사이의 midrid point
                            edge_point_list.append([spot_list[0], spot_list[1], point])
                            check_boolean = True
                            break
                    if not check_boolean:
                        continue
                    else:
                        break

        # calculate max length
        max_length = 0
        max_length_idx = None
        for point_idx, edge_point in enumerate(edge_point_list):
            pt1_e, pt2_e, center_point = edge_point
            # use this code if you draw all each line between two edge point
            # cv2.line(img, pt1_e, pt2_e, color=(255, 255, 0), thickness = 1)
            # cv2.circle(img, pt1_e, radius=2, color=(0, 255, 255), thickness=-1)
            # cv2.circle(img, pt2_e, radius=2, color=(0, 255, 255), thickness=-1)

            length_1 = math.sqrt(math.pow(pt1_e[0] - center_point[0], 2) + math.pow(pt1_e[1] - center_point[1],
                                                                                    2))  # midrid point와 width의 첫 번째 edge point 사이의 거리
            length_2 = math.sqrt(math.pow(center_point[0] - pt2_e[0], 2) + math.pow(center_point[1] - pt2_e[1],
                                                                                    2))  # midrid point와 width의 두 번째 edge point 사이의 거리

            if (length_1 > length_2 * 10) or (length_1 * 10 < length_2):  continue  # # 두 거리의 차이가 심하면 continue

            length = math.sqrt(
                math.pow(pt1_e[0] - pt2_e[0], 2) + math.pow(pt1_e[1] - pt2_e[1], 2))  # length가 가장 높은 것을 선택
            if max_length <= length:
                max_length = length
                max_length_idx = point_idx

        cross_point = None
        if max_length_idx is not None:
            pt1_fe, pt2_fe, cross_point = edge_point_list[max_length_idx]
            self.point_dict["leaf_width_edge_coordinates"].append(pt1_fe)
            self.point_dict["leaf_width_edge_coordinates"].append(pt2_fe)

        else:
            return False, cross_point

        # _slope, _ = get_slope_alpha(pt1_fe[0], pt1_fe[1], pt2_fe[0], pt2_fe[1])
        # print(f"_slope ; {_slope}")

        return True, cross_point

    def find_last_point_midrid(self, leaf_coordinates, width_or_height):
        x_second_last, y_second_last = self.point_dict["midrid_point_coordinate"][-2][0], \
                                       self.point_dict["midrid_point_coordinate"][-2][1]
        x_last, y_last = self.point_dict["midrid_point_coordinate"][-1][0], \
                         self.point_dict["midrid_point_coordinate"][-1][1]

        _slope, _alpha = get_slope_alpha(x_second_last, y_second_last, x_last, y_last)
        # y_last = _slope*x_last + _alpha : (x_last, y_last)에서 기울기 _slope를 가진 1차함수

        continue_check, continue_num, edge_list, skip_num = create_check_flag(leaf_coordinates)

        check_boolean = False
        for i in range(self.margin_error):
            for x, y in leaf_coordinates:  # leaf의 coordinate를 탐색하며 y = _slope*x + _alpha 에 만족하는 y, x coordinate를 찾는다.
                edge_list, continue_num, continue_check = self.find_coordinate_slicing(continue_check, x, y, _slope,
                                                                                       _alpha, edge_list, continue_num,
                                                                                       skip_num, False, i)
                if len(edge_list) == 2:
                    length_1 = math.sqrt(math.pow(edge_list[0][0] - x_last, 2) + math.pow(edge_list[0][1] - y_last, 2))
                    length_2 = math.sqrt(math.pow(edge_list[1][0] - x_last, 2) + math.pow(edge_list[1][1] - y_last, 2))

                    if abs(length_1 - length_2) < (length_1 + length_2) / 3:  # 두 좌표가 모두 기본적으로 x_last, y_last와 멀 경우는 예외
                        edge_list.pop(-1)
                        break

                    check_boolean = True
                    if length_1 <= length_2:
                        self.append_to_leaf_point_dict(edge_list, width_or_height, 0)
                    else:
                        self.append_to_leaf_point_dict(edge_list, width_or_height, 1)
                    break
            if not check_boolean:
                continue
            else:
                break

    def append_to_leaf_point_dict(self, edge_list, width_or_height, num):
        if width_or_height == "width":
            # 이미 midrid의 마지막 point가 새롭게 찾은 edge보다 x값이 더 큰 경우는 pass
            if self.point_dict["midrid_point_coordinate"][-1][0] >= edge_list[num][0]:
                pass
            else:
                self.point_dict["midrid_point_coordinate"].append([edge_list[num][0], edge_list[num][1]])
        else:
            # 이미 midrid의 마지막 point가 새롭게 찾은 edge보다 y값이 더 큰 경우는 pass
            if self.point_dict["midrid_point_coordinate"][-1][1] >= edge_list[num][1]:
                pass
            else:
                self.point_dict["midrid_point_coordinate"].append([edge_list[num][0], edge_list[num][1]])

    def find_first_point_midrid(self, leaf_coordinates, width_or_height):

        x_first, y_first = self.point_dict["midrid_point_coordinate"][0][0], \
                           self.point_dict["midrid_point_coordinate"][0][1]
        x_second, y_second = self.point_dict["midrid_point_coordinate"][1][0], \
                             self.point_dict["midrid_point_coordinate"][1][1]

        _slope, _alpha = get_slope_alpha(x_second, y_second, x_first, y_first)

        continue_check, continue_num, edge_list, skip_num = create_check_flag(leaf_coordinates)

        check_boolean = False
        for i in range(self.margin_error):
            for x, y in leaf_coordinates:  # leaf의 coordinate를 탐색하며 y = _slope*x + _alpha 에 만족하는 y, x coordinate를 찾는다.
                edge_list, continue_num, continue_check = self.find_coordinate_slicing(continue_check, x, y, _slope,
                                                                                       _alpha, edge_list, continue_num,
                                                                                       skip_num, False, i)
                if len(edge_list) == 2:
                    length_1 = math.sqrt(
                        math.pow(edge_list[0][0] - x_first, 2) + math.pow(edge_list[0][1] - y_first, 2))
                    length_2 = math.sqrt(
                        math.pow(edge_list[1][0] - x_first, 2) + math.pow(edge_list[1][1] - y_first, 2))
                    check_boolean = True

                    if abs(length_1 - length_2) < (length_1 + length_2) / 10:
                        edge_list.pop(-1)
                        continue

                    # 두 좌표 중 더 가까운 곳의 좌표를 얻는다.
                    if length_1 <= length_2:
                        self.insert_to_point_dict(edge_list, width_or_height,
                                                  0)  # 긴 line이 length_2일 때, 짧은 부분의 point가 가까운 point
                    else:
                        self.insert_to_point_dict(edge_list, width_or_height,
                                                  1)  # first point가 우측에 있고, 긴 line이 length_1이다.
                    break
            if not check_boolean:
                continue
            else:
                break

    def insert_to_point_dict(self, edge_list, width_or_height, num):
        if width_or_height == "width":
            # 이미 midrid의 첫 번째 point가 새롭게 찾은 edge보다 x값이 더 작은 경우는 pass
            if self.point_dict["midrid_point_coordinate"][-1][0] <= edge_list[num][0]:
                pass
            else:
                self.point_dict["midrid_point_coordinate"].insert(0, [edge_list[num][0], edge_list[num][1]])
        else:
            # 이미 midrid의 첫 번째 point가 새롭게 찾은 edge보다 y값이 더 작은 경우는 pass
            if self.point_dict["midrid_point_coordinate"][-1][1] <= edge_list[num][1]:
                pass
            else:
                self.point_dict["midrid_point_coordinate"].insert(0, [edge_list[num][0], edge_list[num][1]])

    def get_cap_fruit_points(self, outer_idx, inner_info_list, fruit_point_dict):
        # sub_function
        def select_cap_idx(cap_idxs):
            if len(cap_idxs) > 1:  # cap이 2개 이상 포함된 경우
                _inner_idx = None
                min_length = 100000
                for idx in cap_idxs:
                    if idx is None:
                        continue
                    length = get_length(compute_center_point(self.boxes[outer_idx]),
                                        compute_center_point(self.boxes[idx]))
                    if length < min_length:
                        min_length = length
                        _inner_idx = idx
                return _inner_idx
            elif len(cap_idxs) == 1:
                return cap_idxs[0]
            elif len(cap_idxs) == 0:
                return None

        # cap과 cap_2를 분리
        cap_idx, _cap_2_idx = [], []
        for inner_info in inner_info_list:
            inner_idx, inner_objects_name = inner_info
            if 'cap_2' == inner_objects_name:
                _cap_2_idx.append(inner_idx)
            else:
                cap_idx.append(inner_info[0])

        real_cap_idx = select_cap_idx(cap_idx)
        real_cap_2_idx = select_cap_idx(_cap_2_idx)
        if real_cap_idx is not None and real_cap_2_idx is None:
            # cap은 있는데 cap_2가 없는 경우
            fruit_point_dict = self.get_cap_fruit_points_(outer_idx, real_cap_idx, fruit_point_dict,
                                                          is_cap=True, is_cap_2=False)

        elif real_cap_idx is not None and real_cap_2_idx is not None:
            # cap과 cap_2가 둘 다 있는 경우

            fruit_point_dict = self.get_cap_fruit_points_(outer_idx, [real_cap_idx, real_cap_2_idx], fruit_point_dict,
                                                          is_cap=True, is_cap_2=True)
        elif real_cap_idx is None and real_cap_2_idx is None:
            # cap과 cap_2가 둘 다 없는 경우. 이런 경우는 code상 없지만, 만일을 위해 구현
            fruit_only_dict = self.get_only_fruit_points(outer_idx)
            if fruit_only_dict is not None:
                fruit_point_dict["fruit_only"].append(fruit_only_dict)

        elif real_cap_idx is None and real_cap_2_idx is not None:
            # cap은 없는데 cap_2가 있는 경우
            fruit_point_dict = self.get_cap_fruit_points_(outer_idx, real_cap_2_idx, fruit_point_dict,
                                                          is_cap=False, is_cap_2=True)

        return fruit_point_dict

    def get_cap_fruit_points_(self, outer_idx, inner_idx, fruit_point_dict,
                              is_cap, is_cap_2):
        # sub_function  (과의 꼭지가 있을 경우에, 꼭지가 fruit영역 안의 위치에 따라(중앙 or 가장자리) 적용 algorithm이 달라진다.)
        def check_cap_location(inner_idx_):
            # cap이 fruit 영역 안의 중앙 부근에 위치했는지 확인.  (대부분 cap이 fruit의 가장자리 위치에 있음)
            _center_flag = False
            bbox_fruit = self.boxes[outer_idx].copy()
            center_fruit = compute_center_point(bbox_fruit)
            width_fruit, height_fruit = compute_width_height(bbox_fruit)

            center_aredict_fruit = dict(x_min=int(center_fruit[0] - width_fruit / 5),
                                        y_min=int(center_fruit[1] - height_fruit / 5),
                                        x_max=int(center_fruit[0] + width_fruit / 5),
                                        y_max=int(center_fruit[1] + height_fruit / 5))

            center_cap = compute_center_point(self.boxes[inner_idx_])
            center_cap_x, center_cap_y = center_cap

            if center_aredict_fruit['x_min'] < center_cap_x < center_aredict_fruit['x_max'] and \
                    center_aredict_fruit['y_min'] < center_cap_y < center_aredict_fruit['y_max']:
                _center_flag = True
            return _center_flag

        # main
        if is_cap and is_cap_2:
            center_flag = check_cap_location(inner_idx[0])
        else:
            center_flag = check_cap_location(inner_idx)
        fruit_info = dict(segmentation=self.polygons[outer_idx].copy(),
                          type=self.plant_type,
                          bbox=self.boxes[outer_idx])
        if not center_flag:  # cap이 fruit의 center box에 포함되지 않은 경우(가장자리에 위치한 경우)
            # height계산
            fruit_height = self.compute_fruit_height_bokchoy_seed(outer_idx, inner_idx, is_cap, is_cap_2)
            if fruit_height is not None:
                fruit_info['height'] = fruit_height
            if "height" not in list(fruit_info.keys()): return fruit_point_dict
            # width 계산
            fruit_info = self.compute_fruit_width(outer_idx, fruit_info)
            if "width" not in list(fruit_info.keys()): return fruit_point_dict
            fruit_point_dict['cap_fruit_side'].append(fruit_info)
            return fruit_point_dict
        # cap이 fruit의 center box에 포함되는 경우
        # width만 구할 수 있다.
        elif center_flag:
            fruit_info['height'] = list()
            x_center_cap, y_center_cap = compute_center_point(self.boxes[inner_idx])
            fruit_points = self.polygons[outer_idx].copy()
            lowst_lenth = 100000

            # for calculate fruit width
            # length_difference_list : [[length_difference_1, [x_a_1, y_a_1, x_b_1, y_b_1]], [length_difference_2, [x_a_2, y_a_2, x_b_2, y_b_2]], ...]
            length_difference_list = []

            for _fruit_x, _fruit_y in fruit_points:
                ## for calculate fruit width
                _slope, _alpha = get_slope_alpha(_fruit_x, _fruit_y, x_center_cap, y_center_cap)
                # cap의 center coordinate를 지나는 1차함수 위에 존재하는 fruit coordinates를 찾으면 두 점 사이의 거리를 구한 후
                # 기존에 구한 length_1와의 차이값을 length_difference_list에 할당
                # 오차범위 2
                length_1 = get_length([_fruit_x, _fruit_y], [x_center_cap, y_center_cap])
                for x, y in fruit_points:
                    if int(_slope * x + _alpha) - 1 <= int(y) <= int(_slope * x + _alpha) + 1 and (
                            _fruit_x != x and _fruit_y != y):
                        length_2 = get_length([x, y], [x_center_cap, y_center_cap])
                        length_difference_list.append([abs(length_1 - length_2), [_fruit_x, _fruit_y, x, y]])

            for length_difference, coordinates_two_point in length_difference_list:
                if lowst_lenth > length_difference:
                    lowst_lenth = length_difference
                    width_fruit_coordinates = [[coordinates_two_point[0], coordinates_two_point[1]],
                                               [coordinates_two_point[2], coordinates_two_point[3]]]

                    # width의 양 끝점 좌표
            width_points_list = get_width_point(width_fruit_coordinates, 5)

            if len(width_points_list) != 0:
                fruit_info["width"] = width_points_list
                fruit_info["center"] = list(width_points_list[len(width_points_list) // 2])

            if "width" not in fruit_info.keys(): return fruit_point_dict
            fruit_point_dict["cap_fruit_above"].append(fruit_info)
        return fruit_point_dict

    def compute_fruit_height_bokchoy_seed(self, outer_idx, inner_idx, is_cap, is_cap_2):
        # sub_function
        def get_cap_slope(_x_center_cap, _y_center_cap, _cap_points, _fruit_points):
            # cap의 center point 로부터 가장 먼 cap point 찾기
            max_length_cap = -1
            for x, y in _cap_points:
                length_cap = get_length([_x_center_cap, _y_center_cap], [x, y])
                if max_length_cap < length_cap:
                    max_length_cap = length_cap
                    _cap_x, _cap_y = x, y

            # `cap의 center point`와 `cap의 center point 로부터 가장 먼 cap point`사이의 기울기
            slope_cap, _ = get_slope_alpha(_cap_x, _cap_y, _x_center_cap, _y_center_cap)
            if slope_cap == 0:  # 수평인 경우, fruit가 수직으로 똑바로 열려있다고 가정
                # inverse_slope_cap: 위 기울기의 수직인 직선의 기울기
                inverse_slope_cap = 100
            else:
                inverse_slope_cap = (-1 / slope_cap)

            # slope_fruit: `cap의 center point`와 `cap의 center point 로부터 가장 먼 fruit point`사이의 기울기
            max_length_fruit = -1
            for x, y in _fruit_points:
                length_fruit = get_length([_x_center_cap, _y_center_cap], [x, y])
                if max_length_fruit < length_fruit:
                    max_length_fruit = length_fruit
                    _fruit_x, _fruit_y = x, y
            _slope_fruit, _ = get_slope_alpha(_fruit_x, _fruit_y, _x_center_cap, _y_center_cap)

            _slope = compute_median_slope(inverse_slope_cap, _slope_fruit)
            _alpha = _y_center_cap - _x_center_cap * _slope

            return _slope, _alpha

        # sub_function
        def compute_median_slope(_slope_1, _slope_2):
            # slope: 두 기울기의 중간값
            if _slope_1 * _slope_2 > 0:  # 두 기울기가 같은 방향으로 기울어진 경우
                # slope_1와 slope_2의 중간 기울기를 가진 기울기 계산
                _slope = (_slope_1 + _slope_2) / 2
            else:  # 두 기울기가 서로 다른 방향으로 기울어진 경우(대칭처럼)
                # cap의 y좌표에 10을 더한 후, `해당 point`와 `datum_point`사이의 기울기 계산
                # tmp_y = y + 10
                # alpha_fruit = y - slope_2 * x
                # tmp_x_fruit = (tmp_y - alpha_fruit)/slope_2
                # tmp_x_cal = (tmp_y - slope_1_alpha)/slope_1
                # slope, _ = get_slope_alpha(x, y, (tmp_x_cal + tmp_x_fruit)/2, tmp_y)

                # 그냥 절대값 합산
                _slope = abs(_slope_1) + abs(_slope_2)
            return _slope

        # main
        if is_cap and is_cap_2:
            cap_2_idx = inner_idx[1]
            cap_1_idx = inner_idx[0]
            inner_idx = cap_1_idx

        bbox_cap = self.boxes[inner_idx].copy()
        cap_points = self.polygons[inner_idx].copy()
        x_center_cap, y_center_cap = compute_center_point(bbox_cap)
        bbox_fruit = self.boxes[outer_idx].copy()
        fruit_points = self.polygons[outer_idx].copy()

        x_center_fruit, y_center_fruit = compute_center_point(bbox_fruit)
        fruit_bbox_width, fruit_bbox_height = compute_width_height(bbox_fruit)

        if is_cap and is_cap_2:  # cap_1과 cap_2 모두를 포함한 fruit인 경우
            x_center_cap_2, y_center_cap_2 = compute_center_point(self.boxes[cap_2_idx])
            slope_1, alpha_1 = get_slope_alpha(x_center_fruit, y_center_fruit, x_center_cap_2, y_center_cap_2)
            slope_2, _ = get_cap_slope(x_center_cap, y_center_cap, cap_points, fruit_points)
            slope = compute_median_slope(slope_1, slope_2)

            alpha = y_center_cap - x_center_cap * slope
        elif not is_cap and is_cap_2:  # cap_2만을 포함한 fruit인 경우
            slope, alpha = get_slope_alpha(x_center_fruit, y_center_fruit, x_center_cap, y_center_cap)

        else:  # cap만을 포함한 fruit인 경우
            slope, alpha = get_cap_slope(x_center_cap, y_center_cap, cap_points, fruit_points)

        if abs(slope) > 15:  # 기울기가 너무 높은 경우 fruit의 height는 수직의 직선으로 계산한다
            fruit_point_bottom_y, cap_point_bottom_y = -1, -1
            for fruit_x, fruit_y in fruit_points:
                if x_center_cap == fruit_x:
                    if fruit_point_bottom_y < fruit_y:
                        fruit_point_bottom_y = fruit_y
            for cap_x, cap_y in cap_points:
                if x_center_cap == cap_x:
                    if cap_point_bottom_y < cap_y:
                        cap_point_bottom_y = cap_y

            return [[x_center_cap, cap_point_bottom_y], [x_center_cap, fruit_point_bottom_y]]

        elif abs(slope) < 0.067:  # 기울기가 너무 낮은 경우 fruit의 height는 수평의 직선으로 계산한다
            # TODO: cap이 fruit의 좌측에 위치하는지, 우측에 위치하는지 확인 후 case에 따라 code작성
            return None
        else:  # 기울기가 너무 높지 않은 경우
            # cap : fruit의 bottom point, top point를 각각 따로 찾는다.
            # cap_2 : fruit의 bottom point, top point를 한 번에 찾는다.

            # fruit의 bottom point찾기
            edge_list = []
            check_boolean_fruit = False
            is_cap_2_fruit_dict = None
            for i in range(self.margin_error):
                for fruit_x, fruit_y in fruit_points:
                    edge_list, _, _ = self.find_coordinate_slicing(True, fruit_x, fruit_y, slope, alpha, edge_list, 0,
                                                                   0, False, i)

                    if len(edge_list) == 2:
                        if edge_list[0] == edge_list[1]:  # 두 점이 같은 점이면 continue
                            edge_list.pop(-1)
                            continue

                        # 두 점 사이가 fruit_bbox_height/3 보다 낮으면 잘못된 계산, continuie
                        length = get_length(edge_list[0], edge_list[1])
                        if length < fruit_bbox_height / 3:
                            edge_list.pop(-1)
                            continue

                        check_boolean_fruit = True
                        if is_cap_2:
                            # cap_2인 경우 bottom point, top point를 한 번에 저장
                            is_cap_2_fruit_dict = edge_list
                            break

                        # fruit point와 cap의 center point사이의 거리
                        length_1 = get_length(edge_list[0], [x_center_cap, y_center_cap])
                        length_2 = get_length(edge_list[1], [x_center_cap, y_center_cap])

                        if length_1 <= length_2:
                            edge_list.pop(0)
                        else:
                            edge_list.pop(-1)

                        break
                if check_boolean_fruit:
                    break
                else:
                    continue

            if len(edge_list) == 0: return None  # height좌표를 구하지 못한 경우

            if is_cap_2 and is_cap_2_fruit_dict is None:
                # point를 한 개 밖에 찾지 못한 경우
                for fruit_x, fruit_y in fruit_points:
                    # fruit의 하단 point중 x좌표가 비슷한 point를 반대편 point로 결정
                    if 3 > abs(edge_list[0][0] - fruit_x) \
                            and fruit_bbox_height / 3 < get_length(edge_list[0], [fruit_x, fruit_y]):
                        # x좌표의 차이는 3보다 작지만, 두 점의 거리는 fruit_bbox_height/3보다 큰 point
                        edge_list.append([int(fruit_x), int(fruit_y)])
                        break
                is_cap_2_fruit_dict = edge_list

            if is_cap_2 and not is_cap:
                # top, bottom point의 순서대로 tmp_fruit_dict["height"]에 할당
                # center cap point와 각 point사이의 거리가 큰 것이 bottom
                if get_length(is_cap_2_fruit_dict[0], [x_center_cap, y_center_cap]) > \
                        get_length(is_cap_2_fruit_dict[1], [x_center_cap, y_center_cap]):
                    return [is_cap_2_fruit_dict[1], is_cap_2_fruit_dict[0]]
                else:
                    return [is_cap_2_fruit_dict[0], is_cap_2_fruit_dict[1]]

            elif (not is_cap_2 and is_cap) or \
                    (is_cap_2 and is_cap):
                # cap_2는 없고 cap만 있는 경우 or # cap, cap_2 전부 있는 경우
                if fruit_bbox_height / 3 > get_length(edge_list[0], [x_center_cap, y_center_cap]):
                    # fruit의 bottom point와 cap의 center point간의 거리가
                    # fruit_bbox_height/3 보다 작다: fruit의 상단 부근의 point라는 뜻.
                    for fruit_x, fruit_y in fruit_points:
                        # fruit의 하단 point중 x좌표가 동일한 point를 bottom point로 결정
                        if edge_list[0][0] == fruit_x and (fruit_y - edge_list[0][1]) > fruit_bbox_height / 3:
                            edge_list = [[int(fruit_x), int(fruit_y)]]

                fruit_point_bottom = edge_list[0]

                # fruit의 top point를 찾는다.
                # cap인 경우 cap point중에서 top point를 찾아야 한다.
                edge_list_cap = []
                check_boolean_cap = False
                slope_fruit, alpha_fruit = get_slope_alpha(edge_list[0][0], edge_list[0][1], x_center_cap, y_center_cap)
                for i in range(self.margin_error):
                    for cap_x, cap_y in cap_points:
                        edge_list_cap, _, _ = self.find_coordinate_slicing(True, cap_x, cap_y, slope_fruit, alpha_fruit,
                                                                           edge_list_cap, 0, 0, False, i)
                        if len(edge_list_cap) == 2:
                            if edge_list_cap[0] == edge_list_cap[1]:  # 두 point가 같은 값일 경우
                                edge_list_cap.pop(0)
                                continue

                            length = get_length(edge_list_cap[0], edge_list_cap[1])
                            if length < abs(fruit_bbox_height) / 4:  # fruit bbox의 1/4보다 작은 경우: 두 point가 가까이 있을 경우
                                if get_length(edge_list_cap[0], [x_center_fruit, y_center_fruit]) > \
                                        get_length([x_center_fruit, y_center_fruit], [x_center_cap, y_center_cap]):
                                    # 두 점 모두 fruit의 center로부터 cap_center보다 멀리 떨어져있으면 찾는 point가 아니다.
                                    edge_list_cap = []
                                    continue

                            length_1 = get_length(edge_list_cap[0],
                                                  fruit_point_bottom)  # 첫 번째 point와 fruit_bottom point사이의 거리
                            length_2 = get_length(edge_list_cap[1],
                                                  fruit_point_bottom)  # 두 번째 point와 fruit_bottom point사이의 거리
                            check_boolean_cap = True

                            if length_1 <= length_2:
                                edge_list_cap.pop(-1)  # length_2가 크다: edge_list_cap[0]가 더 가까이있는 point == top point
                            else:
                                edge_list_cap.pop(0)
                            break
                    if check_boolean_cap:
                        break
                    else:
                        continue

                if len(edge_list_cap) == 0: return None
                fruit_point_top = edge_list_cap[0]

                return [fruit_point_top, fruit_point_bottom]

    def compute_fruit_width(self, outer_idx, fruit_info):
        fruit_points = self.polygons[outer_idx].copy()

        # height를 표현하는 point 2개의 중점에서의 1차함수 기울기, 절편

        coordinate_1, coordinate_2 = fruit_info["height"]
        [x_1, y_1] = coordinate_1
        [x_2, y_2] = coordinate_2

        # x_mid, y_mid의 위치에서 inverse_slope의 기울기를 가지며 _alpha의 절편을 가진 1차함수 계산
        _slope, _ = get_slope_alpha(x_1, y_1, x_2, y_2)

        if _slope == 0:
            inverse_slope = 20
        elif abs(_slope) > 20:  # 기울기의 절대값이 20보다 크면 폭을 수평으로 계산한다.
            inverse_slope = 0
        else:
            inverse_slope = (-1) / _slope

        mid_point_alpha = []
        mid_point_num = 5
        x_val, y_val = abs(x_1 - x_2) / mid_point_num, abs(y_1 - y_2) / mid_point_num
        num_list = [i - mid_point_num // 2 for i in range(mid_point_num)]

        for i in num_list:
            x_mid, y_mid = (x_1 + x_2) / 2 + (x_val * i), (y_1 + y_2) / 2 + (y_val * i)
            _alpha = y_mid - x_mid * inverse_slope
            mid_point_alpha.append([_alpha, [int(x_mid), int(y_mid)]])

        width_length_list = []
        for mid_point in mid_point_alpha:
            _alpha = mid_point[0]

            _width_coordinates = None
            # fruit boundary coordinates위의 한 coordinate를 구한다.
            check_boolean = False
            for i in range(self.margin_error):
                continue_check, continue_num, edge_list, skip_num = create_check_flag(fruit_points)
                for x, y in fruit_points:
                    edge_list, continue_num, continue_check = self.find_coordinate_slicing(continue_check, x, y,
                                                                                           inverse_slope, _alpha,
                                                                                           edge_list, continue_num,
                                                                                           skip_num, False, i)

                    if len(edge_list) == 2:
                        _width_coordinates = [edge_list[0], edge_list[1]]
                        check_boolean = True
                        break
                if not check_boolean:
                    continue
                else:
                    break

            if _width_coordinates is None: continue
            length = get_length(_width_coordinates[0], _width_coordinates[1])

            width_length_list.append([length, _width_coordinates, mid_point[1]])

        if len(width_length_list) == 0: return fruit_info

        max_length = -1
        for width_info in width_length_list:
            width_length, width_coordi, mid_center_point = width_info

            if max_length < width_length:
                max_length = width_length
                fruit_info["width"] = width_coordi
                fruit_info["center"] = mid_center_point
        return fruit_info

    def get_leaf_width_height(self, bbox_leaf, leaf_coordinate):
        x_min, y_min, x_max, y_max = bbox_leaf
        bbox_width, bbox_height = compute_width_height(bbox_leaf)

        margin_error = self.margin_error
        # leaf_coordinate = np.array(leaf_coordinate)
        # bbox의 세로 길이가 더 긴 경우, 가로축을 엽장으로 측정

        # 세로 길이가 더 긴 경우 세로 길이를 기준으로 1/5지점 분할 후 탐색
        if bbox_height >= bbox_width:
            tmp_points_for_height = [x for x in leaf_coordinate if x[1] <= y_min + bbox_height/5]
            tmp_points_for_width = [x for x in leaf_coordinate if x[0] <= x_min + bbox_width/3]
            # tmp_points_for_height = leaf_coordinate[leaf_coordinate[:,1] <= y_min + bbox_height/4].tolist()
            # tmp_points_for_width = leaf_coordinate[leaf_coordinate[:,0] <= x_min + bbox_width/4].tolist()

        # 가로 길이가 더 긴 경우 가로 길이를 기준으로 1/5지점 분할 후 탐색
        else:
            tmp_points_for_height = [x for x in leaf_coordinate if x[0] <= x_min + bbox_width/5]
            tmp_points_for_width = [x for x in leaf_coordinate if x[1] <= y_min + bbox_height/3]

            # tmp_points_for_height = leaf_coordinate[leaf_coordinate[:,0] <= x_min + bbox_width/4].tolist()
            # tmp_points_for_width = leaf_coordinate[leaf_coordinate[:,1] <= y_min + bbox_height/4].tolist()

        max_height = 0
        heights = []
        # leaf_coordinate = leaf_coordinate.tolist()

        # 세로 또는 가로 1/5 지점에서 모든 points들의 길이 측정 후 가장 긴 길이를 갖는 두 점을 엽장의 두 점으로 설정
        for point1 in tmp_points_for_height:
            for point2 in leaf_coordinate:
                tmp_height = get_length(point1, point2)
                if tmp_height > max_height:
                    max_height = tmp_height
                    heights = [point1, point2]
        try:
            slope, _ = get_slope_alpha(heights[0][0], heights[0][1], heights[1][0], heights[1][1])
        except:
            # heights의 point를 아무것도 찾지 못한 경우 기울기를 0으로 임의 설정 (오류 방지)
            slope = 0

        # 엽장에 수직인 기울기를 엽폭의 기울기로 설정
        if abs(slope) <= 0.1:
            inverse_slope = 10  # heights가 거의 수평인 경우
        else:
            inverse_slope = -1 / slope

        max_width = 0
        widths = []
        # 세로 또는 가로 1/5 지점에서 엽폭의 기울기를 지나는 모든 일차함수 계산 후 최대 길이 반환
        for x, y in tmp_points_for_width:
            alpha = y - x * inverse_slope
            for x2, y2 in leaf_coordinate:
                if x2 == x and y2 == y:
                    continue
                if y2 - margin_error <= int(x2 * inverse_slope + alpha) <= y2 + margin_error:
                    tmp_width = get_length([x,y], [x2,y2])
                    if tmp_width > max_width:
                        widths = [[x,y], [x2,y2]]
                        max_width = tmp_width

        return heights, widths

    def remove_seperate_mask(self, mask, label):
        ### 각 객체별 마스크 중 두개 이상의 mask로 나뉜 객체 중 area가 1000이하인 mask 제거
        new_mask = mask.copy()

        # mask내에 레이블 수 확인
        # 객체 수, 레이블 맵, [x,y,w,h,area], [center_x, center_y]
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)       


        ### 객체 별 mask수가 2개 이하인 경우는 제외 (mask 개수 = 배경 + mask수)
        ### 내부 홀만 채우고 반환
        if num_labels <= 2:
            contours, _ = cv2.findContours(new_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            new_mask = cv2.fillPoly(new_mask, contours, 255)

            x_y_coordi = np.where(new_mask == 255)
            y_min, x_min, y_max, x_max = np.min(x_y_coordi[0]), np.min(x_y_coordi[1]), np.max(x_y_coordi[0]), np.max(x_y_coordi[1])
            new_bbox = [x_min, y_min, x_max, y_max]

            # if self.class_name_list[label] == "midrib" or self.class_name_list[label] == "leaf_width":
            #     cv2.imshow("new mask", new_mask)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()
            return new_mask, new_bbox
        
        stats = stats.tolist()
        sorted_stats = [x.append(idx) for idx, x in enumerate(stats)]
        sorted_stats = sorted(stats, key=lambda x:x[4], reverse=True)

        # 가장 큰 영역은 배경으로 처리
        sorted_stats = sorted_stats[1:]

        # 배경을 제외한 영역중 가장 큰 영역만을 남긴다.
        central_mask = sorted_stats[0]
        new_mask = np.where(labels == central_mask[-1], 1, 0)
        new_mask = new_mask.astype(np.uint8) * 255

        # 윤곽선 검출
        contours, _ = cv2.findContours(new_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 윤곽선 내부 채우기
        new_mask = cv2.fillPoly(new_mask, contours, 255)

        x_y_coordi = np.where(new_mask == 255)
        y_min, x_min, y_max, x_max = np.min(x_y_coordi[0]), np.min(x_y_coordi[1]), np.max(x_y_coordi[0]), np.max(x_y_coordi[1])
        new_bbox = [x_min, y_min, x_max, y_max]

        return new_mask, new_bbox