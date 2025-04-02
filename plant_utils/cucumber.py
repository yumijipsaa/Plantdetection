# plant_type = 'cucumber'
# ### load parts list corresponding to plant type.
# parts_list = json.load(open(
#     os.path.join(os.getcwd(), "for_inference", json.load(open("metadata.json", "r"))[plant_type]["parts_list"]),
#     "r"))
# .
import math

import cv2
import numpy as np

from detectron2.utils.visualizer import GenericMask
from plant_utils.common import *
from utils import get_length, get_slope_alpha, create_check_flag, get_width_point, get_box_degree, get_skeleton_from_mask, get_curve_model, linear_in_fruit, curve_points, curve_angle, rotate_points, get_width_cucumber


class Cucumber:
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
        self.plant_type = 'cucumber'
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
        for outer_objects in outer_objects_cucumber:  # 외부 object를 for문으로 나열
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

                for inner_objects in inner_objects_cucumber:  # 내부 object를 나열
                    for inner_info in self.center_point[inner_objects]:  # 각 내부 object의 중앙점을 나열
                        center_point, inner_idx = inner_info
                        x_center, y_center = center_point

                        # 내부 object의 중앙 점이 외부 object의 영역 안에 위치하는지 확인
                        # 위치하는 경우 외부-내부 object간 대응된다고 판단
                        if x_min - add_area_x_min < x_center < x_max + add_area_x_max and y_min - add_area_y_min < y_center < y_max + add_area_y_max:
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
        for object_name in plant_object_idx_cucumber:
            self.idx_dict[object_name] = []

            # self.set_outer_inner_idx 에서 사용됨
        for object_name in inner_objects_cucumber:
            self.center_point[object_name] = []
        for object_name in outer_objects_cucumber:
            self.bbox[object_name] = []

        # object의 개수를 저장할 dict
        for obj_name in count_object_cucumber:
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
            print(f'checkouter Exist returns false')
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
            for object_name in plant_object_idx_cucumber:
                if self.class_name_list[output] == object_name:
                    print(f'Detected part: {idx} - {object_name}')
                    self.idx_dict[object_name].append(idx)  # 해당 idx를 저장
                    if object_name in inner_objects_cucumber:
                        # inner object는 서로 대응관계가 있는 object인지 확인하기 전 까진 self.useful_mask_idx에 append하지 않는다.
                        self.center_point[object_name].append([compute_center_point(self.boxes[idx]), idx])
                    elif object_name in outer_objects_cucumber:
                        self.bbox[object_name].append([self.boxes[idx], idx])
                        if object_name == "fruit":  # fruit는 대응관계인 cap이 없어도 좌표찾기를 적용할 것이기 때문에 append
                            self.useful_mask_idx.append(idx)
                            # leaf인 경우는 midrid와 대응되지 않는 경우 좌표찾기 적용하지 않을 예정
                    else:
                        self.useful_mask_idx.append(idx)

                    if object_name in count_object_cucumber:
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
        for object_name in result_objects_cucumber:
            if object_name not in SEG_OBJECT_LIST:
                continue
            if idx in self.idx_dict[object_name]:
                if object_name in count_object_cucumber:
                    x_min, y_min, x_max, y_max = self.boxes[idx]
                    self.segmentations[object_name].append([int(x_min), int(y_min), int(x_max), int(y_max)])
                else:
                    self.segmentations[object_name].append(self.polygons[idx].copy())

    # main function
    def calculate_coordinates(self, outputs, img):
        if self.plant_type not in VALID_PLANTS: return None, None, None, None
        predictions = outputs["instances"].to("cpu")
        if not predictions.has("pred_masks"):
            # mask가 한 개도 없으면 save file을 하지 않음
            has_mask = False
            return img, has_mask, None, None
        height = img.shape[0]
        width = img.shape[1]
        self.img = cv2.resize(img, (int(width / self.resize_scale), int(height / self.resize_scale)))

        # 모든 polygon(가장자리 points)를 list에 저장
        self.set_boxes_polygons(height, width, predictions)  # make self.boxes, self.polygons
        self.contain_idx2dict(outputs)  # 모든 object의 index를 dict에 저장
        self.set_outer_inner_idx()  # 서로 대응되는 object의 index를 dict의 key값과 그 안의 list의 요소로 저장

        for idx, _ in enumerate(self.boxes):
            if len(self.polygons[idx]) == 0: continue  # object가 detecting은 됐지만 points(polygon)가 없는 경우도 있다.
            self.set_segmentations_dict(idx)  # self.segmentations에 각 object name별로 index저장

        # 여기서부터 좌표 계산
        coordinates_dict = {}  # 각 계산한 좌표를 저장할 dict
        if len(self.get_object_idx("stem")) != 0:
            coordinates_dict["stem"] = self.get_stem_info()

        # 해당되는 작물들: paprika, cucumber, cucumber, strawberry, chili, chilipepper_seed, cucumber_seed
        if self.check_outer_exist("leaf"):
            coordinates_dict["leaf"] = self.get_draw_leaf_info()
        if self.check_outer_exist("fruit"):
            fruit_point_dict = self.get_draw_fruit_info()
            coordinates_dict["fruit"] = fruit_point_dict

        # object counting
        for object_name in count_object_cucumber:
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

    def get_stem_info(self):
        stem_point_dict_list = []

        ### 하나의 growing에 stem이 두개 이상 매칭되는 것을 방지하기 위한 리스트 생성
        ### predict된 growing 객체 수 만큼 빈 리스트를 생성
        ### 각 리스트에는 매칭 가능한 모든 stem의 인덱스를 담음
        growing_cnt = [[] for _ in range(len(self.idx_dict["growing"]))]

        ### stem의 detection된 인덱스를 담는 리스트
        ### stem_point_dict_list의 stem idx는 달라지기 때문에 리스트 생성
        stem_idx_list = []

        for idx in self.get_object_idx("stem"):
            stem_points = self.polygons[idx].copy()
            bbox_stem = self.boxes[idx].copy()

            stem_info = dict(segmentation=stem_points.copy(),
                             type=self.plant_type,
                             bbox=bbox_stem)

            # tmp_dict 의 key "width", "height", "center" 추가
            stem_info = self.get_stem_points(stem_points,
                                             stem_info,
                                             return_width_or_height(bbox_stem))
            if "width" not in stem_info.keys() or "center" not in stem_info.keys():
                continue
            if "height" not in stem_info.keys():
                continue

            ### 현재 stem 객체가 포함될 수 있는 모든 growing 탐색 후 growing_cnt 리스트의 해당 index에 추가
            growing_cnt = self.check_stem_contatin_growing(stem_points, idx, growing_cnt)

            stem_point_dict_list.append(stem_info)
            stem_idx_list.append(idx)

        ### 하나의 growing에 2개 이상의 stem이 매칭되는 경우 가장 가까운 stem만을 사용
        ### stem_idx_to_find_growing : 최종적으로 matching된 모든 stem의 index list
        stem_idx_to_find_growing = self.check_growing_duplicate(growing_cnt)

        ### stem_idx_to_find_growing에 포함되는 stem만이 True 값을 반환
        for idx in range(len(stem_point_dict_list)):
            if stem_idx_list[idx] in stem_idx_to_find_growing:
                stem_point_dict_list[idx]["contain_growing"] = True
            else:
                stem_point_dict_list[idx]["contain_growing"] = False
        print("total contain growing stem : ", len(set(stem_idx_to_find_growing)))
        return stem_point_dict_list

    def get_stem_points(self, stem_points, stem_info, w_h):
        if w_h == "height":
            # stem_points.sort(key=lambda _x: _x[1])
            stem_coordinates_sorted = sorted(stem_points, key=lambda x: x[1])
        else:
            # stem_points.sort()
            stem_coordinates_sorted = sorted(stem_points)

        # stem_coordinates_sorted = stem_points.copy()
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

            flag1, flag2 = False, False
            for x_stem, y_stem in stem_coordinates_mid_wide:
                if flag1 and flag2:
                    break
                if y_stem == y_bottom and x_stem != x_bottom_right:
                    x_bottom_left = x_stem  # [x_bottom_left, y_bottom] : stem의 3/4지점 좌측 하단 point
                    flag1 = True
                    continue
                if y_stem == y_top and x_stem != x_top_left:
                    x_top_right = x_stem  # [x_top_right, y_top] : stem의 1/4지점 우측 상단 point
                    flag2 = True
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
                if y_right_bottom > y_stem:
                    y_right_bottom = y_stem
                    x_bottom = x_stem
                    # [x_bottom, y_right_bottom] : stem의 3/4지점 우측 하단 point

            y_left_bottom = None
            y_right_top = None

            flag1, flag2 = False, False
            for x_stem, y_stem in stem_coordinates_mid_wide:
                if flag1 and flag2:
                    break
                if x_stem == x_bottom and y_stem != y_right_bottom:
                    # y_left_bottom = y_stem  # [x_bottom, y_left_bottom] : stem의 3/4지점 좌측 하단 point
                    y_right_top = y_stem
                    flag1 = True
                    continue

                if x_stem == x_top and y_stem != y_left_top:
                    # y_right_top = y_stem  # [x_top, y_left_top] : stem의 1/4지점 우측 상단 point
                    y_left_bottom = y_stem
                    flag2 = True
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
                    ### 원본 코드
                    # slope_top = (x_top - x_bottom) / (y_left_top - y_right_top)
                    # slope_bottom = (x_top - x_bottom) / (y_left_bottom - y_right_bottom)

                    # slope_bottom = (x_top - x_bottom) / (y_left_bottom - y_right_bottom)

                    ### 수정 코드
                    slope_top = (y_left_top - y_right_top) / (x_top - x_bottom)
                    slope_bottom = (y_left_bottom - y_right_bottom) / (x_top - x_bottom)

                if slope_bottom * slope_top > 0:
                    slope = (slope_bottom + slope_top) / 2  # stem이 완전한 수평이 아닌, 기울어진 수평일 경우
                else:
                    slope = 0  # stem이 완전한 수평인 경우

            # if abs(slope) <= 0.1:
            #     inverse_slope = 10  # stem이 거의 수평인 경우
            # elif abs(slope) >= 10:
            #     inverse_slope = 0.1  # stem이 거의 수직인 경우
            # else:
            #     inverse_slope = -1 / slope
            #     alpha = -1 * x_center * slope + y_center
            #     inverse_alpha = -1 * x_center * inverse_slope + y_center

            if abs(slope) <= 0.1:
                inverse_slope = 10
            else:
                inverse_slope = -1 / slope
                alpha = -1 * x_center * slope + y_center
                inverse_alpha = -1 * x_center * inverse_slope + y_center

            edge_list = []
            _width_coordinates = None

            ### find width
            ### width의 기울기를 가지며 center point를 지나는 일차함수를 통과하는 point를 탐색한다.

            for x, y in stem_coordinates_mid:
                if abs(inverse_slope) >= 10:
                    if x_center == x:
                        edge_list.append([x, y])
                elif abs(inverse_slope) <= 0.1:
                    if y_center == y:
                        edge_list.append([x, y])
                else:
                    if int(inverse_slope * x + inverse_alpha) - self.margin_error <= int(y) <= int(
                            inverse_slope * x + inverse_alpha) + self.margin_error:
                        edge_list.append([x, y])
                    # if y - self.margin_error <= inverse_slope * x + inverse_alpha  <= y + self.margin_error:
                    #     edge_list.append([x,y])

                # if len(edge_list) == 2:
                #     print("edge list : ", edge_list)
                #     x_1, y_1, x_2, y_2 = edge_list[0][0], edge_list[0][1], edge_list[1][0], edge_list[1][1]

                #     if (w_h == "h" and abs(x_1 - x_2) < box_width/5) or (w_h == "w" and abs(y_1 - y_2) < box_height/5):
                #         edge_list.pop(-1)
                #         continue
                #     if get_length([x_1, y_1], [x_2, y_2]) < 5:
                #         edge_list.pop(-1)
                #         continue
                #     else:
                #         _width_coordinates = edge_list.copy()

            if len(edge_list) == 2:
                _width_coordinates = edge_list

            ## 일차함수를 지나는 점의 개수가 두개 이상이면
            ## 두 점들 중 같은 일차함수와 같은 기울기를 갖는 두 점을 탐색
            elif len(edge_list) > 2:
                flag = False
                for i in range(len(edge_list) - 1):
                    x_1, y_1 = edge_list[i]
                    for j in range(i + 1, len(edge_list)):
                        x_2, y_2 = edge_list[j]

                        if x_1 == x_2:
                            if abs(inverse_slope) >= 10:
                                _width_coordinates = [edge_list[i], edge_list[j]]
                                flag = True
                                break
                            else:
                                continue

                        if abs(inverse_slope) <= 0.1:
                            if y_1 == y_2:
                                _width_coordinates = [edge_list[i], edge_list[j]]
                                flag = True
                                break

                        if inverse_slope - 1 <= (y_1 - y_2) / (x_1 - x_2) <= inverse_slope + 1 \
                                and get_length(edge_list[i], edge_list[j]) >= 5:
                            _width_coordinates = [edge_list[i], edge_list[j]]
                            flag = True
                            break
                    if flag:
                        break

            if _width_coordinates is not None:
                stem_info["width"] = _width_coordinates

                width_1, width_2 = _width_coordinates[0], _width_coordinates[1]
                x_center_point, y_center_point = (width_1[0] + width_2[0]) // 2, (width_1[1] + width_2[1]) // 2
                stem_info["center"] = [x_center_point, y_center_point]

                # stem_coordinates_top_bottom = stem_coordinates_sorted[
                #                               :int(len(stem_coordinates_sorted) / 5)]  # stem의 상단 또는 좌측
                # for coordinate in stem_coordinates_sorted[
                #                   int((len(stem_coordinates_sorted) / 5) * 4):]:  # stem의 하단 또는 우측
                #     stem_coordinates_top_bottom.append(
                #         coordinate)  # stem_coordinates_top_bottom : stem의 양 끝 부분들만 분리

                # for i in range(self.margin_error):
                #     continue_check, continue_num, edge_list, skip_num = create_check_flag(
                #         stem_coordinates_top_bottom)
                #     for x, y in stem_coordinates_top_bottom:
                #         if abs(slope) <= 0.1:
                #             if y_center == y: edge_list.append([x, y])
                #         elif abs(slope) >= 10:
                #             if x_center == x: edge_list.append([x, y])
                #         else:
                #             edge_list, continue_num, continue_check = self.find_coordinate_slicing(continue_check,
                #                                                                                    x, y, slope,
                #                                                                                    alpha, edge_list,
                #                                                                                    continue_num,
                #                                                                                    skip_num, False,
                #                                                                                    i)

                #         if len(edge_list) == 2:
                #             x_1, y_1, x_2, y_2 = edge_list[0][0], edge_list[0][1], edge_list[1][0], edge_list[1][1]
                #             length = get_length([x_1, y_1], [x_2, y_2])

                #             if (w_h == "h" and length < box_height / 4) or (
                #                     w_h == "w" and length < box_width / 4):
                #                 edge_list.pop(-1)
                #                 continue
                #             else:
                #                 stem_info["height"] = edge_list
                #                 break
                #         else:
                #             continue
                #     if "height" not in stem_info.keys():
                #         continue
                #     else:
                #         # 2개의 point였던 height에 center point를 추가
                #         tmp_list = stem_info["height"]
                #         stem_info["height"] = [tmp_list[0], stem_info["center"], tmp_list[1]]
                #         break

                stem_coordinates_top = stem_coordinates_sorted[
                                       :int(len(stem_coordinates_sorted) / 5)]  # stem의 상단 또는 좌측
                stem_coordinates_bottom = stem_coordinates_sorted[
                                          int((len(stem_coordinates_sorted) / 5) * 4):]  # stem의 하단 또는 우측

                # for i in range(self.margin_error):
                #     continue_check_top, continue_num_top, edge_list_top, skip_num_top = create_check_flag(
                #         stem_coordinates_top)
                #     for x, y in stem_coordinates_top:
                #         if abs(slope) <= 0.1:
                #             if y_center == y: edge_list_top.append([x, y])
                #         elif abs(slope) >= 10:
                #             if x_center == x: edge_list_top.append([x, y])
                #         else:
                #             edge_list_top, continue_num_top, continue_check_top = self.find_coordinate_slicing(continue_check_top,
                #                                                                                    x, y, slope,
                #                                                                                    alpha, edge_list_top,
                #                                                                                    continue_num_top,
                #                                                                                    skip_num_top, False,
                #                                                                                    i)

                #     continue_check_bottom, continue_num_bottom, edge_list_bottom, skip_num_bottom = create_check_flag(
                #         stem_coordinates_bottom)
                #     for x, y in stem_coordinates_bottom:
                #         if abs(slope) <= 0.1:
                #             if y_center == y: edge_list_bottom.append([x, y])
                #         elif abs(slope) >= 10:
                #             if x_center == x: edge_list_bottom.append([x, y])
                #         else:
                #             edge_list_bottom, continue_num_bottom, continue_check_bottom = self.find_coordinate_slicing(continue_check_bottom,
                #                                                                                    x, y, slope,
                #                                                                                    alpha, edge_list_bottom,
                #                                                                                    continue_num_bottom,
                #                                                                                    skip_num_bottom, False,
                #                                                                                    i)

                # edge_list_top = sorted(edge_list_top, key=lambda x:get_length(stem_info["center"], x), reverse= True)
                # edge_list_bottom = sorted(edge_list_bottom, key=lambda x:get_length(stem_info["center"], x), reverse= True)

                ### 수정 사항 : 기존 width와 수직인 height를 찾는 알고리즘 -> center point와 가장 먼 point를 찾는 알고리즘으로 변경
                ### stem이 휘어진 경우를 고려하기 위함

                ### height가 수직인 경우 y값이 가장 크거나 작고, x값이 center의 x값과 가장 가까운 두 점을 height으로 설정
                if abs(inverse_slope) <= 0.1:
                    edge_list_top = edge_list_top = sorted(stem_coordinates_top, key=lambda x: (x[1], abs(x_center - x[0])))
                    edge_list_bottom = sorted(stem_coordinates_bottom, key=lambda x: (-x[1], abs(x_center - x[0])))

                ### height가 수평인 경우 x값이 가장 크거나 작고, y값이 center의 y값과 가장 가까운 두 점을 height으로 설정
                elif abs(inverse_slope) >= 10:
                    edge_list_top = sorted(stem_coordinates_top, key=lambda x: (x[0], abs(y_center - x[1])))
                    edge_list_bottom = sorted(stem_coordinates_bottom, key=lambda x: (-x[0], abs(y_center - x[1])))
                else:
                    ### stem의 상단 또는 좌측에서 center point와 가장 멀리 떨어져 있는 점
                    edge_list_top = sorted(stem_coordinates_top, key=lambda x: get_length(stem_info["center"], x), reverse=True)
                    ### stem의 하단 또는 우측에서 center point와 가장 멀리 떨어져 있는 점
                    edge_list_bottom = sorted(stem_coordinates_bottom, key=lambda x: get_length(stem_info["center"], x), reverse=True)

                if edge_list_top and edge_list_bottom:
                    stem_info["height"] = [edge_list_top[0], stem_info["center"], edge_list_bottom[0]]

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
                print(f'Get only fruit .... ')
                fruit_only_dict = self.get_only_fruit_points(outer_idx)
                if fruit_only_dict is not None:
                    fruit_point_dict["fruit_only"].append(fruit_only_dict)
            elif len(inner_info_list) >= 1:  # cap또는 cap_2가 1개 이상 포함된 경우
                print(f'Get cap and fruit .... ')
                fruit_point_dict = self.get_cap_fruit_points(outer_idx, inner_info_list, fruit_point_dict)
        return fruit_point_dict

    def get_only_fruit_points(self, idx):
        fruit_points = self.polygons[idx].copy()
        fruit_points_for_search = self.polygons[idx].copy()
        bbox_fruit = self.boxes[idx]

        fruit_only_info = dict(segmentation=fruit_points.copy(),
                               type=self.plant_type,
                               bbox=bbox_fruit)
        fruit_only_info = self.cucumber_fruit_only(bbox_fruit, fruit_points, fruit_points_for_search, fruit_only_info)
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
                point_dict_list.append(self.point_dict)
            else:
                inner_idx, _ = inner_info_list[0]  # midrid는 각 leaf당 1개만 할당되어있다고 가정

                midrid_points = self.polygons[inner_idx].copy()
                bbox_midrid = self.boxes[inner_idx]
                more_longer = return_width_or_height(bbox_midrid)
                midrid_center_points = get_sorted_center_points(midrid_points, width_or_height=more_longer)
                if len(midrid_center_points) == 0:
                    point_dict_list.append(self.point_dict)
                    continue

                if more_longer:
                    midrid_center_points.sort(key=lambda x: x[1])
                else:
                    midrid_center_points.sort()

                # midrid의 point개수 설정
                num_spot = self.config.NUM_SPOT  # c_cfg.NUM_SPOT == 10 이면 10개의 point를 찍는다.

                count_bot = 1
                ### select particular point
                # midrid point 확보
                for i, center_coordinate in enumerate(midrid_center_points):
                    if i == 0 or i == int(len(midrid_center_points) * (count_bot / num_spot)) or i == (len(midrid_center_points) - 1):
                        count_bot += 1
                        self.point_dict["midrid_point_coordinate"].append(center_coordinate)

                # center point에 더 가까이 있는 midrid가 first midrid point임을 설정한 후 last point를 연장할지, first point를 연장할지 결정
                center_point = int((bbox_leaf[0] + bbox_leaf[2]) / 2), int((bbox_leaf[1] + bbox_leaf[3]) / 2)
                x_first, y_first = self.point_dict["midrid_point_coordinate"][0][0], self.point_dict["midrid_point_coordinate"][0][1]
                x_last, y_last = self.point_dict["midrid_point_coordinate"][-1][0], self.point_dict["midrid_point_coordinate"][-1][1]

                length_from_first_to_center = math.sqrt(math.pow(center_point[0] - x_first, 2) + math.pow(center_point[1] - y_first, 2))
                length_from_last_to_center = math.sqrt(math.pow(center_point[0] - x_last, 2) + math.pow(center_point[1] - y_last, 2))

                # cv2.circle(img, (x_first, y_first), thickness=-1, radius=10, color=(255, 255, 0))
                # cv2.circle(img, (x_last, y_last), thickness=-1, radius=10, color=(255, 0, 255))
                # cv2.circle(img, center_point, thickness=-1, radius=10, color=(0, 0, 255))
                if length_from_first_to_center < length_from_last_to_center:  # first가 center에 더 가까이 있을 경우
                    self.find_last_point_midrid(leaf_points, more_longer)
                else:  # last가 center에 더 가까이 있을 경우
                    self.find_first_point_midrid(leaf_points, more_longer)

                # calculate edge point about width of leaf base on point_coordinate
                check_availability, cross_point = self.find_width_point_midrid(leaf_points, bbox_leaf, more_longer)

                if len(self.point_dict["leaf_width_edge_coordinates"]) == 0:
                    point_dict_list.append(self.point_dict)
                    continue
                self.point_dict["leaf_width_edge_coordinates"] = get_width_point(self.point_dict["leaf_width_edge_coordinates"], 5)

                if cross_point is not None:
                    self.point_dict["center"] = cross_point

                # midrid의 coordinates가 작은 영역에 뭉쳐있다면 point_dict_list에 append하지않음
                first_point, last_point = self.point_dict["midrid_point_coordinate"][0], self.point_dict["midrid_point_coordinate"][-1]
                length = math.sqrt(math.pow(first_point[0] - last_point[0], 2) + math.pow(first_point[1] - last_point[1], 2))

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
            x_min_center_box, x_max_center_box, y_min_center_box, y_max_center_box = x_min, x_max, y_min, int(y_max - height / 4)
        else:
            x_min_center_box, x_max_center_box, y_min_center_box, y_max_center_box = int(x_min + width / 10), int(x_max - width / 4), y_min, y_max

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

            x_before, y_before = self.point_dict["midrid_point_coordinate"][idx - 1][0], self.point_dict["midrid_point_coordinate"][idx - 1][1]
            x_this, y_this = self.point_dict["midrid_point_coordinate"][idx][0], self.point_dict["midrid_point_coordinate"][idx][1]
            x_next, y_next = self.point_dict["midrid_point_coordinate"][idx + 1][0], self.point_dict["midrid_point_coordinate"][idx + 1][1]

            slope_before, _ = get_slope_alpha(x_before, y_before, x_this, y_this)  # midrid의 현재 point와 이전 point사이의 기울기
            slope_next, _ = get_slope_alpha(x_this, y_this, x_next, y_next)  # midrid의 현재 point와 다음 point사이의 기울기

            if slope_before * slope_next < 0:  # 두 기울기의 곱이 음수인 경우 현재 point의 위치를 재조정
                x_npoint, y_npoint = int((x_before + x_next) / 2), int((y_before + y_next) / 2)
                tmp_point_dict.append([x_npoint, y_npoint])
            elif (abs(slope_before) < 1 and abs(slope_next) > 1) or (abs(slope_before) > 1 and abs(slope_next) < 1):  # 기울기가 급격하게 변하는 경우 현재 point의 위치를 재조정
                x_npoint, y_npoint = int((x_before + x_next) / 2), int((y_before + y_next) / 2)
                tmp_point_dict.append([x_npoint, y_npoint])
            else:
                tmp_point_dict.append(self.point_dict["midrid_point_coordinate"][idx])

        self.point_dict["midrid_point_coordinate"] = tmp_point_dict

        edge_point_list = []
        for idx in range(len(self.point_dict["midrid_point_coordinate"])):
            if idx == 0 or idx == len(self.point_dict["midrid_point_coordinate"]) - 1:
                continue

            x_this, y_this = self.point_dict["midrid_point_coordinate"][idx][0], self.point_dict["midrid_point_coordinate"][idx][1]
            x_next, y_next = self.point_dict["midrid_point_coordinate"][idx + 1][0], self.point_dict["midrid_point_coordinate"][idx + 1][1]

            if not (x_min_center_box < x_this < x_max_center_box and y_min_center_box < y_this < y_max_center_box):
                continue  # 현재 점이 center box안에 위치하지 않는 경우
            _slope, _ = get_slope_alpha(x_this, y_this, x_next, y_next)

            # -----TODO : for 박람회,       point평균값 적용    # 2022-12-12
            # if (idx < 2) or (idx > len(self.point_dict["midrid_point_coordinate"])-3):  _slope, _ = get_slope_alpha(x_this, y_this, x_next, y_next)
            # else :
            #     # 앞 뒤 각 2개의 point에 대한 기울기를 계산 후 평균값 적용
            #     _slope_sum = 0
            #     for num, i in enumerate(range(idx -2, idx + 2)):
            #         x_tmp_this, y_tmp_this = self.point_dict["midrid_point_coordinate"][i][0], self.point_dict["midrid_point_coordinate"][i][1]
            #         x_tmp_next, y_tmp_next = self.point_dict["midrid_point_coordinate"][i+1][0], self.point_dict["midrid_point_coordinate"][i+1][1]
            #         tmp_slope, _ = get_slope_alpha(x_tmp_this, y_tmp_this, x_tmp_next, y_tmp_next)
            #         _slope_sum += tmp_slope
            #     _slope = _slope_sum/(num+1)
            # -----

            # _slope, _ = get_slope_alpha(x_this, y_this, x_next, y_next)   # 롤백

            if _slope == 0:
                continue  # 기울기가 0인경우는 잘못 계산된 것

            # -----TODO : for 박람회        # 2022-12-12
            # if self.plant_type in ["cucumber", "cucumber", "strawberry"] :
            #     if (abs(_slope) > 20): _slope = 100
            #     elif (abs(_slope) < 1/20) : _slope = 1/100
            #     elif (midrid_slope * _slope < 0) : continue

            # else:
            #     if (abs(midrid_slope) > 20) or (abs(_slope) > 20): _slope = 100
            #     elif (abs(midrid_slope) < 1/20) or (abs(_slope) < 1/20) : _slope = 1/100
            #     elif (midrid_slope * _slope < 0) : continue
            # -----

            # if (abs(midrid_slope) > 10) or (abs(_slope) > 20): pass # _slope = 100             # midrid의 기울기 (또는 point의 기울기)가 너무 높으면 잎이 수직으로 찍힌 경우. width를 수평으로 긋는다.
            # elif (abs(midrid_slope) < 1/10) or (abs(_slope) < 1/20) : pass # _slope = 1/100        # midrid의 기울기 (또는 point의 기울기)가 너무 낮으면 잎이 수평으로 찍힌 경우. width를 수직으로 긋는다.
            # elif (midrid_slope * _slope < 0) : continue       # 두 기울기는 서로 음수, 양수 관계여야 한다. 단, midrid_slope가 너무 높거나 낮은 경우는 제외

            if _slope == 0:
                print("slope : 0")
                continue
            else:
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
                        spot_list, continue_num, continue_check = self.find_coordinate_slicing(continue_check, x, y,
                                                                                               inverse_slope, _alpha,
                                                                                               spot_list, continue_num,
                                                                                               skip_num,
                                                                                               False, i)
                        if len(spot_list) == 2:
                            length = math.sqrt(math.pow(spot_list[0][0] - spot_list[1][0], 2) + math.pow(
                                spot_list[0][1] - spot_list[1][1], 2))
                            if length < 20:
                                spot_list.pop(-1)
                                continue

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
        x_second_last, y_second_last = self.point_dict["midrid_point_coordinate"][-2][0], self.point_dict["midrid_point_coordinate"][-2][1]
        x_last, y_last = self.point_dict["midrid_point_coordinate"][-1][0], self.point_dict["midrid_point_coordinate"][-1][1]

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

        cap_idx = []
        for inner_info in inner_info_list:
            cap_idx.append(inner_info[0])

        if cap_idx[0] is not None:
            fruit_point_dict = self.get_cap_fruit_points_(outer_idx, cap_idx[0], fruit_point_dict, is_cap=True)
        return fruit_point_dict

    def get_cap_fruit_points_(self, outer_idx, inner_idx, fruit_point_dict, is_cap):

        fruit_info = dict(segmentation=self.polygons[outer_idx].copy(),
                          type=self.plant_type,
                          bbox=self.boxes[outer_idx])

        # height계산
        # fruit_height, fruit_width, fruit_center = self.cucumber_fruit_cap(outer_idx, inner_idx)
        # if fruit_height is not None:
        #     # fruit_info['height'], fruit_info['width'], fruit_info['center'] = fruit_height, fruit_width, fruit_center
        #     fruit_info['width'], fruit_info['center'] = fruit_width, fruit_center

        fruit_points = self.polygons[outer_idx].copy()
        height, curve, curved_points = self.is_curved(fruit_points)

        fruit_points = np.asarray([np.asarray(i) for i in fruit_points])
        mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
        mask = cv2.fillPoly(mask, [fruit_points], (255, 255, 255))
        width = get_width_cucumber(height, mask)

        fruit_info['height'] = height
        fruit_info['width'] = width
        fruit_info['center'] = [math.floor((width[0][0] + width[1][0]) / 2), math.floor((width[0][1] + width[1][1]) / 2)]
        fruit_info['is_curved'] = curve
        fruit_info['curved_points'] = curved_points

        if "height" not in list(fruit_info.keys()):
            return fruit_point_dict

        # width 계산
        if "width" not in list(fruit_info.keys()):
            return fruit_point_dict

        fruit_point_dict['cap_fruit_side'].append(fruit_info)
        return fruit_point_dict

    def cucumber_fruit_cap(self, outer_idx, inner_idx):
        bbox_fruit = self.boxes[outer_idx]
        fruit_points = self.polygons[outer_idx].copy()
        fruit_points_for_search = fruit_points.copy()
        if return_width_or_height(bbox_fruit) == "height":  # width < length
            # sort according to y coordinates
            fruit_points.sort(key=lambda x: x[1])
        else:  # length < width  : 옆으로 누운 것
            # sort according to x coordinates
            fruit_points.sort()

        mid_points = get_cucumber_mid_points(fruit_points, 10, True)

        mid_points_ = mid_points[int(len(mid_points) * (1 / 4)):]
        first_point_x, first_point_y = mid_points_[0]
        cap_center_x, cap_center_y = compute_center_point(self.boxes[inner_idx])
        slope, alpha = get_slope_alpha(first_point_x, first_point_y, cap_center_x, cap_center_y)

        cap_points = self.polygons[inner_idx].copy()
        img = self.img  ###

        check_boolean = False
        cap_edge_list = []
        for i in range(self.margin_error):
            for x, y in cap_points:
                if abs(slope) > 4:
                    if return_width_or_height(bbox_fruit) == "height":
                        if cap_center_x == x:
                            cap_edge_list.append([x, y])
                    else:
                        if cap_center_y == y:
                            cap_edge_list.append([x, y])
                else:
                    cap_edge_list, _, _ = self.find_coordinate_slicing(True, x, y, slope, alpha, cap_edge_list, 1, 1,
                                                                       False, i)

                if len(cap_edge_list) == 2:

                    if get_length(cap_edge_list[0], cap_edge_list[1]) < 5:
                        cap_edge_list.pop(-1)
                        continue
                    check_boolean = True
                    break

            if not check_boolean:
                continue
            else:
                break

        if len(cap_edge_list) < 2:
            height_points = mid_points
        else:
            min_length = 100000
            first_point = None
            for point in cap_edge_list:
                length = get_length(compute_center_point(bbox_fruit), point)
                if length < min_length:
                    min_length = length
                    first_point = point

            mid_points_.insert(0, first_point)
            height_points = mid_points_

        max_length = -1
        width_points = None
        for i in range(len(height_points)):
            if i >= int(len(height_points) * (4 / 5)) or i <= int(len(height_points) * (1 / 5)):
                continue  # 양 끝 1/5개수 만큼의 point는 탐색하지 않는다
            point_1_x, point_1_y = height_points[i]
            point_2_x, point_2_y = height_points[i + 1]
            center_point_x, center_point_y = compute_center_point([point_1_x, point_1_y, point_2_x, point_2_y])
            slope, alpha = get_slope_alpha(point_1_x, point_1_y, point_2_x, point_2_y)
            if slope == 0:
                continue
            inverse_slope = (-1) / slope
            inverse_alpha = center_point_y - center_point_x * inverse_slope

            tmp_width_points = None
            check_boolean = False
            for i in range(self.margin_error):
                continue_check, continue_num, width_edge_list, skip_num = create_check_flag(fruit_points_for_search)
                for x, y in fruit_points_for_search:
                    width_edge_list, continue_num, continue_check = self.find_coordinate_slicing(continue_check, x, y,
                                                                                                 inverse_slope,
                                                                                                 inverse_alpha,
                                                                                                 width_edge_list,
                                                                                                 continue_num, skip_num,
                                                                                                 False, i)

                    if len(width_edge_list) == 2:
                        tmp_width_points = [width_edge_list[0], width_edge_list[1]]
                        check_boolean = True
                        break
                if not check_boolean:
                    continue
                else:
                    break

            if tmp_width_points is None:
                continue

            width_length = get_length(tmp_width_points[0], tmp_width_points[1])
            if width_length > max_length:
                max_length = width_length
                width_points = tmp_width_points

        if width_points is None:
            return None, None
        else:
            width_1, width_2 = width_points
            center_points = compute_center_point([width_1[0], width_1[1], width_2[0], width_2[1]])
            return height_points, width_points, center_points

    def compute_fruit_w_h_common_fruit(self, outer_idx, inner_idx):  # sub function
        def get_top_point(fruit_points_part_1, x_center_cap, y_center_cap):
            min_length = 10000
            top_point = None
            for point in fruit_points_part_1:  # cap의 center point로부터 가장 가까운 fruit point가 top point
                tmp_length = get_length([x_center_cap, y_center_cap], point)
                if min_length > tmp_length:
                    top_point = point
                    min_length = tmp_length
            return top_point

        # sub function
        def get_bottom_point(fruit_points_part_2):
            bottom_point = None
            # 가장 낮은 y좌표들 중 x 기준 중점을 bottom point로 사용
            bottom_point_list = []
            max_y_coordinate = -1
            for point in fruit_points_part_2:
                if max_y_coordinate <= point[1]:
                    max_y_coordinate = point[1]
                    bottom_point_list.append(point)

            if len(bottom_point_list) == 1:
                bottom_point = bottom_point_list[0]
            elif len(bottom_point_list) >= 2:
                bottom_point_list.sort()
                bottom_point = bottom_point_list[len(bottom_point_list) // 2]

            return bottom_point

        # main
        fruit_info = dict()

        x_center_cap, y_center_cap = compute_center_point(self.boxes[inner_idx])

        fruit_bbox_width, fruit_bbox_height = compute_width_height(self.boxes[outer_idx])
        fruit_points = self.polygons[outer_idx].copy()
        fruit_points.sort(key=lambda x: x[1])

        fruit_points_part_1 = fruit_points[:int(len(fruit_points) / 3)]
        fruit_points_part_2 = fruit_points[int(len(fruit_points) * 2 / 3):]

        top_point = get_top_point(fruit_points_part_1, x_center_cap, y_center_cap)
        bottom_point = get_bottom_point(fruit_points_part_2)

        if top_point is None or bottom_point is None:
            return None

        # fruit_info["height"] = [top_point, bottom_point]
        width_center_point = [int((top_point[0] + bottom_point[0]) / 2), int((top_point[1] + bottom_point[1]) / 2)]
        fruit_info["center"] = width_center_point

        slope, _ = get_slope_alpha(top_point[0], top_point[1], bottom_point[0], bottom_point[1])

        # ----
        width_point = []
        inverse_slope = -1 / (slope)
        inverse_alpha = -1 * width_center_point[0] * inverse_slope + width_center_point[1]
        check_boolean = False
        for i in range(3):
            continue_check, _, edge_list, skip_num = create_check_flag(fruit_points)
            continue_num = self.margin_error + 10
            for x, y in fruit_points:
                edge_list, continue_num, continue_check = self.find_coordinate_slicing(continue_check, x, y,
                                                                                       inverse_slope, inverse_alpha,
                                                                                       edge_list, continue_num,
                                                                                       skip_num, False, i)

                if len(edge_list) == 2:
                    fruit_length = math.sqrt(
                        math.pow(edge_list[0][0] - edge_list[1][0], 2) + math.pow(edge_list[0][1] - edge_list[1][1], 2))
                    if fruit_length < fruit_bbox_width / 2:  # 두 점 사이의 거리가 짧으면 continue(최소 fruit bbox의 절반은 되어야 한다.)
                        edge_list.pop(-1)
                        continue
                    width_point = [edge_list[0], edge_list[1]]
                    check_boolean = True
                    break
            if not check_boolean:
                continue
            else:
                break

        if len(width_point) == 0:
            width_point = []
            fruit_coordinates_part_3 = fruit_points[int(len(fruit_points) * 1 / 3):int(len(fruit_points) * 2 / 3)]
            for point in fruit_coordinates_part_3:
                if point[1] == width_center_point[1]:
                    if len(width_point) == 0:
                        width_point.append(point)
                        continue
                    elif len(width_point) == 1:
                        half_horizon_length = math.sqrt(math.pow(width_center_point[0] - point[0], 2) + math.pow(width_center_point[1] - point[1], 2))
                        horizon_length = math.sqrt(math.pow(width_point[0][0] - point[0], 2) + math.pow(width_point[0][1] - point[1], 2))

                        if half_horizon_length * 1.5 < horizon_length:
                            width_point.append(point)
                            break

        width_coordinates = width_point

        # ----

        # ---TODO : for 박람회
        # if abs(slope) >= 30:  # fruit의 length선의 기울기가 20 이상인 경우 width의 기울기 == 0
        #     fruit_coordinates_part_3 = fruit_coordinates[int(len(fruit_coordinates)*1/3):int(len(fruit_coordinates)*2/3)]

        #     for point in fruit_coordinates_part_3:
        #         if point[1] == width_center_point[1]:
        #             if len(width_point) == 0:
        #                 width_point.append(point)
        #                 continue
        #             elif len(width_point) == 1:
        #                 half_horizon_length = math.sqrt(math.pow(width_center_point[0] - point[0], 2) + math.pow(width_center_point[1] - point[1], 2))
        #                 horizon_length = math.sqrt(math.pow(width_point[0][0] - point[0], 2) + math.pow(width_point[0][1] - point[1], 2))

        #                 if half_horizon_length * 1.5 < horizon_length:
        #                     width_point.append(point)
        #                     break

        #     width_coordinates = width_point
        # else:
        #     inverse_slope = -1/(slope)
        #     inverse_alpha = -1*width_center_point[0]*inverse_slope + width_center_point[1]
        #     # fruit boundary coordinates위의 한 coordinate를 구한다.

        #     check_boolean = False
        #     for i in range(self.margin_error):
        #         continue_check, _, edge_list, skip_num = create_check_flag(fruit_coordinates)
        #         continue_num = self.margin_error + 10
        #         for x, y in fruit_coordinates:
        #             edge_list, continue_num, continue_check = self.find_coordinate_slicing(continue_check, x, y, inverse_slope, inverse_alpha, edge_list, continue_num, skip_num, False, i)

        #             if len(edge_list) == 2:
        #                 fruit_length = math.sqrt(math.pow(edge_list[0][0] - edge_list[1][0], 2) + math.pow(edge_list[0][1] - edge_list[1][1], 2))
        #                 if fruit_length < fruit_bbox_width/2 :    # 두 점 사이의 거리가 짧으면 continue(최소 fruit bbox의 절반은 되어야 한다.)
        #                     edge_list.pop(-1)
        #                     continue
        #                 width_point = [edge_list[0], edge_list[1]]
        #                 check_boolean = True
        #                 break
        #         if not check_boolean: continue
        #         else : break

        #     width_coordinates = width_point
        # ----

        fruit_info["width"] = width_coordinates

        if "width" not in fruit_info.keys() or "height" not in fruit_info.keys():
            return None
        return fruit_info

    def cucumber_fruit_only(self, bbox_fruit, fruit_points, fruit_points_for_search, fruit_only_info):
        # fruit_points_copy = fruit_points.copy()
        # if return_width_or_height(bbox_fruit) == "height":  # width < length
        #     # sort according to y coordinates
        #     fruit_points.sort(key=lambda x: x[1])
        #     fruit_points_for_search.sort()
        # else:  # length < width  : 옆으로 누운 것
        #     # sort according to x coordinates
        #     fruit_points.sort()
        #     fruit_points_for_search.sort(key=lambda x: x[1])
        #
        # mid_points_cucumber = get_cucumber_mid_points(fruit_points, 10, True)
        #
        # fruit_only_info["height"] = mid_points_cucumber
        # mid_point_idx = int(len(mid_points_cucumber) / 2)
        # # mid_points_cucumber 중 mid point 를 기준으로 -1번째 point와 +1번째 point사이의 기울기 계산
        # for i, point in enumerate(mid_points_cucumber):
        #     if i == mid_point_idx:
        #         center_before_point, center_after_point = mid_points_cucumber[i - 1], mid_points_cucumber[i + 1]
        #         break
        # x_1, y_1 = center_before_point
        # x_2, y_2 = center_after_point
        # slope, _ = get_slope_alpha(x_1, y_1, x_2, y_2)
        #
        # if slope == 0:
        #     inverse_slope = 20
        # else:
        #     inverse_slope = (-1) / slope
        #
        # # compute width
        # x_min, _, x_max, _ = bbox_fruit
        # center_point = compute_center_point(bbox_fruit)
        # x_mid, y_mid = center_point[0], center_point[1]
        # alpha = y_mid - x_mid * (inverse_slope)
        #
        # width_coordinates = None
        # check_boolean = False
        # for i in range(self.margin_error):
        #     continue_check, continue_num, edge_list, skip_num = create_check_flag(fruit_points_for_search)
        #     for x, y in fruit_points_for_search:
        #         edge_list, continue_num, continue_check = self.find_coordinate_slicing(continue_check, x, y,
        #                                                                                inverse_slope, alpha, edge_list,
        #                                                                                continue_num, skip_num, False, i)
        #
        #         if len(edge_list) == 2:
        #             width_coordinates = [edge_list[0], edge_list[1]]
        #             check_boolean = True
        #             break
        #     if not check_boolean:
        #         continue
        #     else:
        #         break
        #
        # if width_coordinates is None:
        #     return None
        #
        # fruit_only_info["width"] = width_coordinates
        #
        # # 가장 높은 point와 가장 낮은 point의 수직 거리 계산
        # first_point, last_point = fruit_only_info['height'][0], fruit_only_info['height'][-1]
        #
        # fruit_only_info["depth_point"] = dict()
        # fruit_only_info["depth_point"]["first_point"] = first_point
        # fruit_only_info["depth_point"]["last_point"] = last_point
        #
        # fruit_only_info["vertical_point"] = dict()
        # fruit_width = int(x_max - x_min)
        # if first_point[0] < last_point[0]:  # 오른쪽으로 휘었을 때
        #     if (self.img.shape[1] // 2) < first_point[0]:  # 화면 우즉에 있을 때
        #         fruit_only_info["vertical_point"]["first_point"] = (first_point[0] - fruit_width, first_point[1])
        #         fruit_only_info["vertical_point"]["last_point"] = (first_point[0] - fruit_width, last_point[1])
        #     else:  # 화면 좌측에 있을 때
        #         fruit_only_info["vertical_point"]["first_point"] = (last_point[0] + fruit_width, first_point[1])
        #         fruit_only_info["vertical_point"]["last_point"] = (last_point[0] + fruit_width, last_point[1])
        # else:  # 왼쪽으로 휘었을 때
        #     if (self.img.shape[1] // 2) > last_point[0]:  # 화면 좌측에 있을 때
        #         fruit_only_info["vertical_point"]["first_point"] = (first_point[0] + fruit_width, first_point[1])
        #         fruit_only_info["vertical_point"]["last_point"] = (first_point[0] + fruit_width, last_point[1])
        #     else:  # 화면 우즉에 있을 때
        #         fruit_only_info["vertical_point"]["first_point"] = (first_point[0] - fruit_width, first_point[1])
        #         fruit_only_info["vertical_point"]["last_point"] = (first_point[0] - fruit_width, last_point[1])
        #
        # if "width" not in fruit_only_info.keys() or "height" not in fruit_only_info.keys():
        #     return None
        #
        # #오이의 center는 width point의 중점
        # width_1, width_2 = fruit_only_info["width"][0], fruit_only_info["width"][1]
        # fruit_only_info['center'] = [(width_1[0] + width_2[0]) // 2, (width_1[1] + width_2[1]) // 2]

        height, curve, curved_points = self.is_curved(fruit_points)
        fruit_points = np.asarray([np.asarray(i) for i in fruit_points])
        mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
        mask = cv2.fillPoly(mask, [fruit_points], (255, 255, 255))
        width = get_width_cucumber(height, mask)

        fruit_only_info['height'] = height
        fruit_only_info['width'] = width
        fruit_only_info['center'] = [math.floor((width[0][0] + width[1][0]) / 2), math.floor((width[0][1] + width[1][1]) / 2)]
        fruit_only_info['is_curved'] = curve
        fruit_only_info['curved_points'] = curved_points

        return fruit_only_info

    def is_curved(self, seg):
        seg = np.asarray([np.asarray(i) for i in seg])
        blank = np.zeros(self.img.shape[:2], dtype=np.uint8)
        mask_img = cv2.fillPoly(blank, [seg], (255, 255, 255))

        # 최소 바운딩 박스
        angle, center = get_box_degree(seg)

        # 계산된 각도로 회전, 회전이미지에서 윤곽선 추출
        M = cv2.getRotationMatrix2D((math.floor(center[0]), math.floor(center[1])), angle, 1.0)
        mask_img = cv2.warpAffine(mask_img, M, (self.img.shape[1], self.img.shape[0]))

        # 1280 * n 사이즈로 리사이즈
        size_to_x = 1280
        size_to_y = self.img.shape[0] * size_to_x / self.img.shape[1]
        mask_img_temp = cv2.resize(mask_img, (size_to_x, round(size_to_y)))

        # 스켈레톤 좌표 그룹 리턴
        x, y = get_skeleton_from_mask(mask_img_temp)

        # 원래 이미지 사이즈에 해당하는 좌표로 변환
        for i in range(len(x)):
            x[i], y[i] = x[i] * self.img.shape[1] / size_to_x, y[i] * self.img.shape[1] / size_to_x

        # 선형회귀
        model_y = get_curve_model(x, y, 2)
        predicts, y_range = linear_in_fruit(model_y, mask_img, 2)

        model_midline = get_curve_model(x, y, 3)
        predicts_mid, y_range_mid = linear_in_fruit(model_midline, mask_img, 3)

        l = curve_points(model_y, predicts, y_range)
        a = curve_angle(predicts, y_range)
        mid_points = [[x, y[0]] for (x, y) in zip(predicts_mid, y_range_mid)]

        curved = []
        if abs(a) > 15:  # 사잇각이 임계값을 넘을 경우 곡과로 판단
            curved.append(True)
        else:
            curved.append(False)

        l = rotate_points(l, angle, center)
        mid_points = rotate_points(mid_points, angle, center)

        points = []
        for i in range(len(l)):
            point_dict = {'x': l[i][0], 'y': l[i][1]}
            points.append(point_dict)

        return mid_points, curved, points

    def check_stem_contatin_growing(self, stem_points, stem_idx, growing_cnt):
        ### stem 객체가 포함 될 수 있는 growing 객체를 모두 탐색 후 list에 stem 인덱스를 추가하는 함수
        ### 인식된 모든 growing 탐색
        for i, idx in enumerate(self.idx_dict["growing"]):
            growing_bbox = self.boxes[idx].copy()  # growing bbox
            x_min, y_min, x_max, y_max = growing_bbox
            flag = False

            # stem point가 growing bbox 안에 포함되면 True 반환
            # y 축 오차 범위 (- 50 ~ + 50), x축 오차 범위 (-10 ~ +10)
            for x, y in stem_points:
                if x_min - 10 <= x <= x_max + 10 and y_min - 50 <= y <= y_max + 50:
                    ### 인식된 growing 객체의 index에 stem의 index를 추가 (중복 방지)
                    growing_cnt[i].append(stem_idx)
                    flag = True
                if flag:
                    break
        return growing_cnt

    def check_growing_duplicate(self, growing_cnt):
        ### 하나의 growing 객체에 stem이 여러개 인식되는 것을 방지
        origin_growing_idx = self.idx_dict["growing"]
        ### 중복 제거 후 각 growing 객체 당 하나의 stem idx만을 반환
        stem_idx_to_change_true = []

        for idx, growing in enumerate(growing_cnt):
            if len(growing) >= 2:
                # 매칭된 growing의 bbox
                growing_bbox = self.boxes[origin_growing_idx[idx]]
                # growing bbox의 center point
                growing_x, growing_y = int(growing_bbox[0] + growing_bbox[2]) // 2, int(growing_bbox[1] + growing_bbox[3]) // 2
                min_length = 999999

                # 매칭된 stem의 index를 저장할 변수
                true_idx = None

                for i in growing:
                    # stem의 bbox
                    stem_bbox = self.boxes[i]
                    # stem bbox의 center point
                    stem_x, stem_y = int(stem_bbox[0] + stem_bbox[2]) // 2, int(stem_bbox[1] + stem_bbox[3]) // 2
                    # 매칭된 growing과의 거리 계산
                    tmp_length = get_length([growing_x, growing_y], [stem_x, stem_y])

                    # 가장 가까운 stem의 index 탐색
                    if min_length > tmp_length:
                        min_length = tmp_length
                        true_idx = i

                stem_idx_to_change_true.append(true_idx)

            # growing에 매칭된 stem이 한개인 경우에는 해당 stem을 매칭
            elif len(growing) == 1:
                stem_idx_to_change_true.append(growing[0])

        return stem_idx_to_change_true
