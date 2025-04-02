# plant_type = 'tomato'
# ### load parts list corresponding to plant type.
# parts_list = json.load(open(
#     os.path.join(os.getcwd(), "for_inference", json.load(open("metadata.json", "r"))[plant_type]["parts_list"]),
#     "r"))
# .

import cv2
import numpy as np

from detectron2.utils.visualizer import GenericMask
from plant_utils.common import *
from utils import (get_length, get_slope_alpha, create_check_flag, get_width_point)

from scipy.spatial.distance import cdist


class Tomato:
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
        self.plant_type = 'tomato'
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
        for outer_objects in outer_objects_tomato:  # 외부 object를 for문으로 나열
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
                if outer_objects == "fruit":
                    # 포함 area를 height의 1/4만큼 올린다. (cap은 보통 위에 위치)
                    # image는 위가 y값의 시작이다
                    add_area_y_min = +height / 4
                    add_area_y_max = -height / 4

                for inner_objects in inner_objects_tomato:  # 내부 object를 나열
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
        for object_name in plant_object_idx_tomato:
            self.idx_dict[object_name] = []

            # self.set_outer_inner_idx 에서 사용됨
        for object_name in inner_objects_tomato:
            self.center_point[object_name] = []
        for object_name in outer_objects_tomato:
            self.bbox[object_name] = []

        # object의 개수를 저장할 dict
        for obj_name in count_object_tomato:
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
        for idx, output in enumerate(outputs["instances"]["pred_classes"]):

            for object_name in plant_object_idx_tomato:
                if self.class_name_list[output] == object_name:
                    print(f'Detected part: {idx} - {object_name}')
                    self.idx_dict[object_name].append(idx)  # 해당 idx를 저장

                    if object_name in inner_objects_tomato:
                        # inner object는 서로 대응관계가 있는 object인지 확인하기 전 까진 self.useful_mask_idx에 append하지 않는다.
                        self.center_point[object_name].append([compute_center_point(self.boxes[idx]), idx])
                    elif object_name in outer_objects_tomato:
                        self.bbox[object_name].append([self.boxes[idx], idx])
                        if object_name == "fruit":  # fruit는 대응관계인 cap이 없어도 좌표찾기를 적용할 것이기 때문에 append
                            self.useful_mask_idx.append(idx)
                            # leaf인 경우는 midrid와 대응되지 않는 경우 좌표찾기 적용하지 않을 예정
                        if object_name == "leaf":  # fruit는 대응관계인 cap이 없어도 좌표찾기를 적용할 것이기 때문에 append
                            self.useful_mask_idx.append(idx)
                    else:
                        self.useful_mask_idx.append(idx)

                    if object_name in count_object_tomato:
                        self.count_object[object_name] += 1

    def set_boxes_polygons(self, height, width, predictions):
        boxes = predictions["pred_boxes"] if predictions.get("pred_boxes") else None

        for box in boxes:
            self.boxes.append([int(box[0]), int(box[1]), int(box[2]), int(box[3])])

        masks = np.asarray(predictions["pred_masks"])
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
        for object_name in result_objects_tomato:
            if object_name not in SEG_OBJECT_LIST:
                continue
            if idx in self.idx_dict[object_name]:
                if object_name in count_object_tomato:
                    x_min, y_min, x_max, y_max = self.boxes[idx]
                    self.segmentations[object_name].append([int(x_min), int(y_min), int(x_max), int(y_max)])
                else:
                    self.segmentations[object_name].append(self.polygons[idx].copy())

    # main function
    def calculate_coordinates(self, outputs, img):
        if self.plant_type not in VALID_PLANTS:
            return None, None, None, None
        # predictions = outputs["instances"].to("cpu")
        predictions = outputs["instances"]
        
        
        for idx, mask in enumerate(predictions['pred_masks']):
            mask = np.array(mask).astype(np.uint8) * 255
            n_mask, n_bbox = self.remove_seperate_mask(mask, predictions["pred_classes"][idx])
            predictions['pred_masks'][idx] = n_mask
            predictions['pred_boxes'][idx] = n_bbox

            if self.class_name_list[predictions["pred_classes"][idx]] in ["midrib", "leaf_width"]:
                self.extract_skeleton(predictions['pred_masks'][idx])
                predictions['pred_masks'][idx] = self.get_skeleton(predictions['pred_masks'][idx])
                
        
        if len(predictions.get("pred_masks")) < 1:
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
            if len(self.polygons[idx]) == 0:
                continue  # object가 detecting은 됐지만 points(polygon)가 없는 경우도 있다.
            self.set_segmentations_dict(idx)  # self.segmentations에 각 object name별로 index저장

        # 여기서부터 좌표 계산
        coordinates_dict = {}  # 각 계산한 좌표를 저장할 dict

        if len(self.get_object_idx("stem")) != 0:
            coordinates_dict["stem"] = self.get_stem_info()
        if self.check_outer_exist("leaf"):
            coordinates_dict["leaf"] = self.get_draw_leaf_info()
        if self.check_outer_exist("fruit"):
            fruit_point_dict = self.get_draw_fruit_info()
            coordinates_dict["fruit"] = fruit_point_dict
        # object counting
        for object_name in count_object_tomato:
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

    def get_draw_leaf_info(self):
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
                ### midrid와 leaf_width 분리
                midrid_idx = [x[0] for x in inner_info_list if x[1] == "midrib"]
                width_idx = [x[0] for x in inner_info_list if x[1] == "leaf_width"]

                midrid_idx = midrid_idx[0] if len(midrid_idx) > 0 else None
                width_idx = width_idx[0] if len(width_idx) > 0 else None

                mask_img = np.zeros((self.img.shape[0], self.img.shape[1]), np.uint8)
                mask_img2 = mask_img.copy()


                if midrid_idx is not None:
                    midrid_points = self.polygons[midrid_idx].copy()
                    bbox_midrid = self.boxes[midrid_idx]
                    more_longer = return_width_or_height(bbox_midrid)
                    # midrid_center_points = get_sorted_center_points(midrid_points, width_or_height=more_longer)
                    midrid_center_points = midrid_points
                    
                    for x,y in midrid_center_points:
                        cv2.circle(mask_img, (x,y), radius=3, thickness=-1, color=(255,255,255))
                    # cv2.imshow("midrid center point", mask_img)
                    # cv2.waitKey()
                    # cv2.destroyAllWindows()

                    if len(midrid_center_points) == 0:
                        point_dict_list.append(point_dict_list)
                        continue
                    
                    if more_longer == "height":
                        midrid_center_points.sort(key=lambda x: x[1], reverse=True)
                    else:
                        midrid_center_points.sort(key = lambda x:x[0])

                    
                
                    # midrid의 point개수 설정
                    num_spot = self.config.NUM_SPOT  # c_cfg.NUM_SPOT == 10 이면 10개의 point를 찍는다.
                    count_bot = 1
                    ### select particular point
                    # midrid point 확보
                    for i, center_coordinate in enumerate(midrid_center_points):
                        if i == 0 or \
                                i == int(len(midrid_center_points) * (count_bot / num_spot)) or \
                                i == (len(midrid_center_points) - 1):
                            count_bot += 1
                            self.point_dict["midrid_point_coordinate"].append(center_coordinate)


                    # center point에 더 가까이 있는 midrid가 first midrid point임을 설정한 후 last point를 연장할지, first point를 연장할지 결정
                    center_point = int((bbox_leaf[0] + bbox_leaf[2]) / 2), int((bbox_leaf[1] + bbox_leaf[3]) / 2)
                    x_first, y_first = self.point_dict["midrid_point_coordinate"][0][0], \
                                       self.point_dict["midrid_point_coordinate"][0][1]
                    x_last, y_last = self.point_dict["midrid_point_coordinate"][-1][0], \
                                     self.point_dict["midrid_point_coordinate"][-1][1]

                    length_from_first_to_center = math.sqrt(
                        math.pow(center_point[0] - x_first, 2) + math.pow(center_point[1] - y_first, 2))
                    length_from_last_to_center = math.sqrt(
                        math.pow(center_point[0] - x_last, 2) + math.pow(center_point[1] - y_last, 2))
                    
                    # cv2.circle(img, (x_first, y_first), thickness=-1, radius=10, color=(255, 255, 0))
                    # cv2.circle(img, (x_last, y_last), thickness=-1, radius=10, color=(255, 0, 255))
                    # cv2.circle(img, center_point, thickness=-1, radius=10, color=(0, 0, 255))
                    # print(self.point_dict["midrid_point_coordinate"])
                    
                    


                    # if length_from_first_to_center < length_from_last_to_center:  # first가 center에 더 가까이 있을 경우
                    #     self.find_last_point_midrid(leaf_points, more_longer, "midrib")
                    # else:  # last가 center에 더 가까이 있을 경우
                    #     self.find_first_point_midrid(leaf_points, more_longer, "midrib")


                    # calculate edge point about width of leaf base on point_coordinate
                    # check_availability, cross_point = self.find_width_point_midrid(leaf_points, bbox_leaf, more_longer)
                    # self.find_first_point(self.point_dict["midrid_point_coordinate"])
                    # init_path = self.nearest_neighbor(self.point_dict["midrid_point_coordinate"], 0)
                    # print(f"init path : {init_path}")
                    # optimized_path = self.two_opt(self.point_dict["midrid_point_coordinate"], init_path)
                    # print(optimized_path)
                    # path_points = [self.point_dict["midrid_point_coordinate"][x] for x in optimized_path]
                    # self.point_dict["midrid_point_coordinate"] = path_points
                    # self.point_dict["midrid_point_coordinate"] = self.sort_midrib_points(self.point_dict["midrid_point_coordinate"])
                    path = self.find_first_point(self.point_dict["midrid_point_coordinate"])
                    self.point_dict["midrid_point_coordinate"] = [self.point_dict["midrid_point_coordinate"][x] for x in path]


                    # for x,y in self.point_dict["midrid_point_coordinate"]:
                    #     cv2.circle(mask_img, (x,y), radius=3, thickness=-1, color=(255,255,255))
                    # cv2.imshow("zero img", mask_img)
                    # cv2.waitKey()
                    # cv2.destroyAllWindows()

                    if len(self.point_dict["leaf_width_edge_coordinates"]) == 0 and width_idx is None:
                        point_dict_list.append(self.point_dict)
                        continue
                    
                    # self.point_dict["leaf_width_edge_coordinates"] = get_width_point(
                    #     self.point_dict["leaf_width_edge_coordinates"], 5)


                    # if cross_point is not None:
                    #     self.point_dict["center"] = cross_point

                    # midrid의 coordinates가 작은 영역에 뭉쳐있다면 point_dict_list에 append하지않음
                    first_point, last_point = self.point_dict["midrid_point_coordinate"][0], \
                                              self.point_dict["midrid_point_coordinate"][-1]
                    length = math.sqrt(math.pow(first_point[0] - last_point[0], 2) + math.pow(first_point[1] - last_point[1], 2))
                    
                    box_width, box_height = compute_width_height(bbox_midrid)

                    check_availability = True
                    if more_longer == "width":
                        if length < box_width / 7:
                            check_availability = False
                    elif more_longer == "height":
                        if length < box_height / 7:
                            check_availability = False


                    if check_availability == False:
                        self.point_dict["midrid_point_coordinate"] = []
                    point_dict_list.append(self.point_dict)


                if width_idx is not None:
                    print("width idx is Not None")
                    width_points = self.polygons[width_idx].copy()
                    bbox_width = self.boxes[width_idx]
                    more_longer = return_width_or_height(bbox_width)
                    width_center_points = get_sorted_center_points(width_points, width_or_height=more_longer)

                    if len(width_center_points) == 0:
                        print("width center points == 0")
                        point_dict_list.append(point_dict_list)
                        continue

                    if more_longer == "height":
                        width_center_points.sort(key=lambda x: x[1])
                    else:
                        width_center_points.sort()
                    
                    # midrid의 point개수 설정
                    num_spot = self.config.NUM_SPOT  # c_cfg.NUM_SPOT == 10 이면 10개의 point를 찍는다.
                    count_bot = 1
                    ### select particular point
                    # midrid point 확보
                    for i, center_coordinate in enumerate(width_center_points):
                        if i == 0 or \
                                i == int(len(width_center_points) * (count_bot / num_spot)) or \
                                i == (len(width_center_points) - 1):
                            count_bot += 1
                            self.point_dict["leaf_width_edge_coordinates"].append(center_coordinate)

                    # center point에 더 가까이 있는 midrid가 first midrid point임을 설정한 후 last point를 연장할지, first point를 연장할지 결정
                    center_point = int((bbox_leaf[0] + bbox_leaf[2]) / 2), int((bbox_leaf[1] + bbox_leaf[3]) / 2)
                    x_first, y_first = self.point_dict["leaf_width_edge_coordinates"][0][0], \
                                       self.point_dict["leaf_width_edge_coordinates"][0][1]
                    x_last, y_last = self.point_dict["leaf_width_edge_coordinates"][-1][0], \
                                     self.point_dict["leaf_width_edge_coordinates"][-1][1]

                    length_from_first_to_center = math.sqrt(
                        math.pow(center_point[0] - x_first, 2) + math.pow(center_point[1] - y_first, 2))
                    length_from_last_to_center = math.sqrt(
                        math.pow(center_point[0] - x_last, 2) + math.pow(center_point[1] - y_last, 2))
                    
                    # cv2.circle(img, (x_first, y_first), thickness=-1, radius=10, color=(255, 255, 0))
                    # cv2.circle(img, (x_last, y_last), thickness=-1, radius=10, color=(255, 0, 255))
                    # cv2.circle(img, center_point, thickness=-1, radius=10, color=(0, 0, 255))


                    if length_from_first_to_center < length_from_last_to_center:  # first가 center에 더 가까이 있을 경우
                        self.find_last_point_midrid(leaf_points, more_longer, "width")
                    else:  # last가 center에 더 가까이 있을 경우
                        self.find_first_point_midrid(leaf_points, more_longer, "width")

                    # calculate edge point about width of leaf base on point_coordinate
                    check_availability, cross_point = self.find_width_point_midrid(leaf_points, bbox_leaf, more_longer)

                    # if len(self.point_dict["leaf_width_edge_coordinates"]) == 0:
                    #     point_dict_list.append(self.point_dict)
                    #     continue
                    
                    # self.point_dict["leaf_width_edge_coordinates"] = get_width_point(
                    #     self.point_dict["leaf_width_edge_coordinates"], 5)


                    if cross_point is not None:
                        self.point_dict["center"] = cross_point

                    # midrid의 coordinates가 작은 영역에 뭉쳐있다면 point_dict_list에 append하지않음
                    first_point, last_point = self.point_dict["leaf_width_edge_coordinates"][0], \
                                              self.point_dict["leaf_width_edge_coordinates"][-1]
                    length = math.sqrt(math.pow(first_point[0] - last_point[0], 2) + math.pow(first_point[1] - last_point[1], 2))
                    
                    box_width, box_height = compute_width_height(bbox_width)
                    
                    # if more_longer == "width":
                    #     if length < box_width / 7:
                    #         check_availability = False
                    # elif more_longer == "height":
                    #     if length < box_height / 7:
                    #         check_availability = False
                    check_availability = True
                    if check_availability:
                        point_dict_list.append(self.point_dict)       

        return point_dict_list
    
    def find_last_point_midrid(self, leaf_coordinates, width_or_height, midrib_or_leaf_width):
        if midrib_or_leaf_width == "midrib":
            x_second_last, y_second_last = self.point_dict["midrid_point_coordinate"][-2][0], \
                                        self.point_dict["midrid_point_coordinate"][-2][1]
            x_last, y_last = self.point_dict["midrid_point_coordinate"][-1][0], \
                            self.point_dict["midrid_point_coordinate"][-1][1]
        else:
            x_second_last, y_second_last = self.point_dict["leaf_width_edge_coordinates"][-2][0], \
                                        self.point_dict["leaf_width_edge_coordinates"][-2][1]
            x_last, y_last = self.point_dict["leaf_width_edge_coordinates"][-1][0], \
                            self.point_dict["leaf_width_edge_coordinates"][-1][1]

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
                        self.append_to_leaf_point_dict(edge_list, width_or_height, 0, midrib_or_leaf_width)
                    else:
                        self.append_to_leaf_point_dict(edge_list, width_or_height, 1, midrib_or_leaf_width)
                    break
                
            if not check_boolean:
                continue
            else:
                break

    def find_first_point_midrid(self, leaf_coordinates, width_or_height, midrib_or_leaf_width):

        if midrib_or_leaf_width == "midrib":
            x_first, y_first = self.point_dict["midrid_point_coordinate"][0][0], \
                            self.point_dict["midrid_point_coordinate"][0][1]
            x_second, y_second = self.point_dict["midrid_point_coordinate"][1][0], \
                                self.point_dict["midrid_point_coordinate"][1][1]
        else:
            x_first, y_first = self.point_dict["leaf_width_edge_coordinates"][0][0], \
                            self.point_dict["leaf_width_edge_coordinates"][0][1]
            x_second, y_second = self.point_dict["leaf_width_edge_coordinates"][1][0], \
                                self.point_dict["leaf_width_edge_coordinates"][1][1]

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
                        self.insert_to_point_dict(edge_list, width_or_height,0, midrib_or_leaf_width)  # 긴 line이 length_2일 때, 짧은 부분의 point가 가까운 point
                    else:
                        self.insert_to_point_dict(edge_list, width_or_height,1, midrib_or_leaf_width)  # first point가 우측에 있고, 긴 line이 length_1이다.
                    break
                
            if not check_boolean:
                continue
            else:
                break
    def insert_to_point_dict(self, edge_list, width_or_height, num, midrib_or_leaf_width):
        if midrib_or_leaf_width == "midrib":
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
        else:
            if width_or_height == "width":
                # 이미 midrid의 첫 번째 point가 새롭게 찾은 edge보다 x값이 더 작은 경우는 pass
                if self.point_dict["leaf_width_edge_coordinates"][-1][0] <= edge_list[num][0]:
                    pass
                else:
                    self.point_dict["leaf_width_edge_coordinates"].insert(0, [edge_list[num][0], edge_list[num][1]])
            else:
                # 이미 midrid의 첫 번째 point가 새롭게 찾은 edge보다 y값이 더 작은 경우는 pass
                if self.point_dict["leaf_width_edge_coordinates"][-1][1] <= edge_list[num][1]:
                    pass
                else:
                    self.point_dict["leaf_width_edge_coordinates"].insert(0, [edge_list[num][0], edge_list[num][1]])


    def append_to_leaf_point_dict(self, edge_list, width_or_height, num, midrib_or_leaf_width):
        if midrib_or_leaf_width == "midrib":
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

        else:
            if width_or_height == "width":
                # 이미 midrid의 마지막 point가 새롭게 찾은 edge보다 x값이 더 큰 경우는 pass
                if self.point_dict["leaf_width_edge_coordinates"][-1][0] >= edge_list[num][0]:
                    pass
                else:
                    self.point_dict["leaf_width_edge_coordinates"].append([edge_list[num][0], edge_list[num][1]])
            else:
                # 이미 midrid의 마지막 point가 새롭게 찾은 edge보다 y값이 더 큰 경우는 pass
                if self.point_dict["leaf_width_edge_coordinates"][-1][1] >= edge_list[num][1]:
                    pass
                else:
                    self.point_dict["leaf_width_edge_coordinates"].append([edge_list[num][0], edge_list[num][1]])


    def find_width_point_midrid(self, leaf_coordinates, bbox_leaf, width_or_height):
        # calculate center box coordinates
        x_min, y_min, x_max, y_max = bbox_leaf
        width, height = x_max - x_min, y_max - y_min

        if width_or_height == "height":
            x_min_center_box, x_max_center_box, y_min_center_box, y_max_center_box = x_min, x_max, y_min, int(
                y_max - height / 4)
        else:
            x_min_center_box, x_max_center_box, y_min_center_box, y_max_center_box = x_min, x_max, y_min, y_max

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



            # _slope, _ = get_slope_alpha(x_this, y_this, x_next, y_next)   # 롤백

            if _slope == 0: continue  # 기울기가 0인경우는 잘못 계산된 것

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

            if (length_1 > length_2 * 10) or (length_1 * 10 < length_2):continue  # # 두 거리의 차이가 심하면 continue

            length = math.sqrt(
                math.pow(pt1_e[0] - pt2_e[0], 2) + math.pow(pt1_e[1] - pt2_e[1], 2))  # length가 가장 높은 것을 선택
            if max_length <= length:
                max_length = length
                max_length_idx = point_idx

        cross_point = None
        if max_length_idx is not None:
            pt1_fe, pt2_fe, cross_point = edge_point_list[max_length_idx]
            # self.point_dict["leaf_width_edge_coordinates"].append(pt1_fe)
            # self.point_dict["leaf_width_edge_coordinates"].append(pt2_fe)

        else:
            return False, cross_point

        # _slope, _ = get_slope_alpha(pt1_fe[0], pt1_fe[1], pt2_fe[0], pt2_fe[1])
        # print(f"_slope ; {_slope}")

        return True, cross_point
    
    def get_tomato_leaf_coordinates(self):
        """
            tomato의 leaf는 midrid를 기준으로 height와 width를 계산
        """

        # sub function
        def except_points_in_bbox(bbox, point_list):
            x_min, y_min, x_max, y_max = bbox
            tmp_list = []
            for point in point_list:
                if x_min - 10 < point[0] and point[0] < x_max + 10 and y_min - 10 < point[1] and point[1] < y_max + 10:
                    continue
                tmp_list.append(point)

            return tmp_list

        # sub function
        def get_midpoint_in_three_point(point_list, distance, first_point=None, last_point=None, img=None):
            """
            point_list를 순서대로 탐색하며 다음 점과의 거리, 다다음 점과의 거리가 크게 차이가 없을 때 세 점의 중점을 append한다.
            다음 점과의 거리는 distance를 shreshold로 한다.
            """
            next_point_list = []
            continue_count = 2
            for i, point_I in enumerate(point_list):
                if first_point == True and i == 0:
                    next_point_list.append(point_I)
                    continue
                if last_point == True and i == len(point_list) - 1:
                    next_point_list.append(point_I)
                    break

                if continue_count != 0:
                    continue_count -= 1
                    continue

                if last_point == True and i == len(point_list) - 2:
                    next_point_list.append(point_I)
                    next_point_list.append(point_list[i + 1])
                    break
                elif i + 1 >= len(point_list) - 1:
                    next_point_list.append(point_I)
                    break

                point_II = point_list[i + 1]
                point_III = point_list[i + 2]

                length_II = math.sqrt(math.pow(point_I[0] - point_II[0], 2) + math.pow(point_I[1] - point_II[1], 2))
                length_III = math.sqrt(math.pow(point_I[0] - point_III[0], 2) + math.pow(point_I[1] - point_III[1], 2))
                if length_II < distance and length_II * 1.7 > length_III:
                    point = [int((point_I[0] + point_II[0] + point_III[0]) / 3),
                             int((point_I[1] + point_II[1] + point_III[1]) / 3)]
                    next_point_list.append(point)
                    continue_count = 2
                else:
                    next_point_list.append(point_I)
            return next_point_list

        # sub function
        def get_nearest_point(last_point, points_list):
            min_length = 100000

            for point in points_list:
                length = math.sqrt(math.pow(last_point[0] - point[0], 2) + math.pow(last_point[1] - point[1], 2))
                if min_length > length:
                    first_point = point
                    min_length = length

            return first_point

        # sub function
        def combind_two_list(previous_list, next_list, last_point=None, img=None):
            """
            conbind next_list to previous_list by append, in order is nearest length
            """
            exist_points = []
            min_length = 100000
            last_point_previous = previous_list[-1]

            # previous_list의 last point로부터 nearest한 next_list point를 탐색
            first_point_next_list = get_nearest_point(last_point_previous, next_list)
            previous_list.append(first_point_next_list)
            exist_points.append(first_point_next_list)

            # 위에서 구한 first_point_next_list point을 기준으로, next_list 탐색하며 가까운 순으로 append
            for _ in next_list:
                min_length = 100000
                near_point = previous_list[-1]
                for point in next_list:
                    if point in exist_points: continue

                    length = math.sqrt(math.pow(near_point[0] - point[0], 2) + math.pow(near_point[1] - point[1], 2))
                    if min_length > length:
                        min_length = length
                        nearest_point = point

                if last_point is not None:
                    if last_point == nearest_point:
                        exist_points.append(nearest_point)
                        previous_list.append(nearest_point)
                        break
                exist_points.append(nearest_point)
                previous_list.append(nearest_point)

            return previous_list

        # sub function
        def rearange_last_midrid_coordinates(_last_midrid_coordinates):
            divided_last_midrid_coordinates_list = [_last_midrid_coordinates[:int(len(_last_midrid_coordinates) / 3)],
                                                    _last_midrid_coordinates[int(len(_last_midrid_coordinates) / 3):int(
                                                        len(_last_midrid_coordinates) * 2 / 3)],
                                                    _last_midrid_coordinates[int(len(_last_midrid_coordinates) * 2 / 3):]]
            tmp_list = []

            for last_midrid_coordinates_part in divided_last_midrid_coordinates_list:
                tmp_dict = {}
                x_min_tmp, y_min_tmp, x_max_tmp, y_max_tmp = 10000, 10000, -1, -1
                for part_point in last_midrid_coordinates_part:
                    if x_min_tmp > part_point[0]: x_min_tmp = part_point[0]
                    if x_max_tmp < part_point[0]: x_max_tmp = part_point[0]
                    if y_min_tmp > part_point[1]: y_min_tmp = part_point[1]
                    if y_max_tmp < part_point[1]: y_max_tmp = part_point[1]

                tmp_dict["width"] = x_max_tmp - x_min_tmp
                tmp_dict["height"] = y_max_tmp - y_min_tmp
                tmp_dict["center"] = [(x_max_tmp + x_min_tmp) // 2, (y_max_tmp + y_min_tmp) // 2]
                tmp_list.append(tmp_dict)

            for i, dict_ in enumerate(tmp_list):
                if i == 0 or i == 1:
                    if dict_["width"] < dict_["height"]:
                        if tmp_list[i + 1]["center"][1] > dict_["center"][1]:
                            # 다음 y center가 더 높은 값이면 오름차순 y sort
                            divided_last_midrid_coordinates_list[i].sort(key=lambda x: x[1])
                        else:
                            divided_last_midrid_coordinates_list[i].sort(key=lambda x: x[1], reverse=True)
                    else:
                        if tmp_list[i + 1]["center"][0] > dict_["center"][0]:
                            # 다음 x center가 더 높은 값이면 오름차순 x sort
                            divided_last_midrid_coordinates_list[i].sort()
                        else:
                            divided_last_midrid_coordinates_list[i].sort(reverse=True)
                else:
                    if dict_["width"] < dict_["height"]:
                        if tmp_list[i - 1]["center"][1] > dict_["center"][1]:
                            # 이전 y center가 더 높은 값이면 내림차순 y sort
                            divided_last_midrid_coordinates_list[i].sort(key=lambda x: x[1], reverse=True)
                        else:
                            divided_last_midrid_coordinates_list[i].sort(key=lambda x: x[1])
                    else:
                        if tmp_list[i - 1]["center"][0] > dict_["center"][0]:
                            # 이전 x center가 더 높은 값이면 내림차순 x sort
                            divided_last_midrid_coordinates_list[i].sort(reverse=True)
                        else:
                            divided_last_midrid_coordinates_list[i].sort()

            _last_midrid_coordinates = []
            for last_midrid_coordinates_part in divided_last_midrid_coordinates_list:
                _last_midrid_coordinates += last_midrid_coordinates_part
                
            return _last_midrid_coordinates

        # main function
        point_dict_list = []
        for outer_idx, inner_info_list in self.outer_inner_idx['leaf'].items():
            bbox_midrid = self.boxes[int(outer_idx)]
            tomato_leaf_info = dict(segmentation=self.polygons[int(outer_idx)].copy(),
                                    type=self.plant_type,
                                    bbox=bbox_midrid)

            # first midrid 또는 last midrid중 하나라도 없는지 확인
            # first midrid 또는 last midrid중 하나라도 없는 경우는 continue
            first_midrid_flag, last_midrid_flag = False, False
            for inner_info in inner_info_list:
                _, inner_object_name = inner_info
                if inner_object_name == "first_midrid":
                    first_midrid_flag = True
                elif inner_object_name == "last_midrid":
                    last_midrid_flag = True
            if not (first_midrid_flag and last_midrid_flag):
                point_dict_list.append(tomato_leaf_info)
                continue

            outer_idx = int(outer_idx)
            midrid_dict = dict()
            midrid_dict["midrid"] = dict(points=self.polygons[outer_idx].copy(),
                                         bbox=bbox_midrid)
            midrid_dict["mid_midrid"] = list()

            for inner_info in inner_info_list:
                inner_idx, inner_object_name = inner_info

                if inner_object_name == "mid_midrid":
                    midrid_dict[inner_object_name].append(compute_center_point(self.boxes[inner_idx]))

                for object in ["first_midrid", "last_midrid"]:
                    if inner_object_name == object:
                        midrid_dict[object] = dict(
                            points=self.polygons[inner_idx].copy(),
                            bbox=self.boxes[inner_idx])

            # midrid의 center points를 계산
            midrid_center_points = get_tomato_center_points(midrid_dict['midrid']["points"],
                                                                 object="midrid",
                                                                 width_or_height=None,
                                                                 mid_midrid_center_points=midrid_dict["mid_midrid"])
            # mid_midrid의 center points까지 append
            for center_x_y in midrid_dict["mid_midrid"]:
                midrid_center_points.append(center_x_y)

            ### phase_1
            # first_midrid가 midrid의 center보다 왼쪽에 있으면 midrid_center_points를 x 오름차순 sort
            # 그 반대는 x 내림차순 sort
            x_center_first, y_center_first = compute_center_point(midrid_dict["first_midrid"]["bbox"])
            x_center_midrid = compute_center_point(bbox_midrid)[0]
            if x_center_midrid > x_center_first:
                midrid_center_points.sort()
            else:
                midrid_center_points.sort(reverse=True)
            midrid_points_phase_1 = midrid_center_points

            # combine first_midrid_coordinates and midmidrid_and_midrid_coordinates to point_list_phase_1
            # first_midrid영역 안에 있는 midmidrid_and_midrid_coordinates를 제외
            midrid_points_phase_1 = except_points_in_bbox(midrid_dict["first_midrid"]["bbox"], midrid_points_phase_1)

            # 가까이 모여있는 세 개의 point를 하나의 point로 치환
            midrid_points_phase_1 = get_midpoint_in_three_point(midrid_points_phase_1,
                                                                10,
                                                                first_point=True,
                                                                last_point=False)

            # get center coordinates of first_midrid and sort forward to right direction
            first_midrid_cneter_points = get_tomato_center_points(midrid_dict["first_midrid"]['points'],
                                                                       object="first",
                                                                       width_or_height=return_width_or_height(
                                                                           midrid_dict["first_midrid"]["bbox"]))

            if return_width_or_height(midrid_dict["first_midrid"]["bbox"]) == "height":  # first_midrid의 세로길이가 클 때
                # first_midrid_coordinates의 y center coordinates가 midrid_points_phase_1 첫 번째 point의 y coordinate보다 낮은 곳에 위치한다면
                # first_midrid_coordinates를 y기준 오름차순 sort(image상으로는 아래가 높은 값)
                # 그 반대는 y기준 내림차순 sort
                if y_center_first < midrid_points_phase_1[0][1]:
                    first_midrid_cneter_points.sort(key=lambda x: x[1])
                else:
                    first_midrid_cneter_points.sort(key=lambda x: x[1], reverse=True)
            else:  # first_midrid의 가로길이가 클 때
                if x_center_midrid > x_center_first:
                    first_midrid_cneter_points.sort()
                else:
                    first_midrid_cneter_points.sort(reverse=True)

            ### phase_2
            midrid_points_phase_2 = combind_two_list(first_midrid_cneter_points, midrid_points_phase_1)

            # get center coordinates of last_midrid and sort forward to right direction
            # last_midrid 에 대해서 x 또는 y sort
            last_midrid_coordinates = get_tomato_center_points(midrid_dict["last_midrid"]['points'],
                                                                    object="last",
                                                                    width_or_height=return_width_or_height(
                                                                        midrid_dict["last_midrid"]["bbox"]))
            # last_midrid_영역 안에 있는 point_list_phase_2를 제외
            point_list_phase_2 = except_points_in_bbox(midrid_dict["last_midrid"]["bbox"], midrid_points_phase_2)

            print(f'point_list_phase_2: {point_list_phase_2}')
            if len(point_list_phase_2) == 0:
                point_dict_list.append(tomato_leaf_info)
                continue

            nearest_point_of_last_midrid = get_nearest_point(point_list_phase_2[-1], last_midrid_coordinates)

            x_center_last, y_center_last = compute_center_point(midrid_dict["last_midrid"]["bbox"])

            if return_width_or_height(midrid_dict["first_midrid"]["bbox"]) == "height":
                if y_center_last < nearest_point_of_last_midrid[1]:
                    # midrid에 가장 가까운 point가 last midrid의 ceneter y coordinate보다 높으면(아래 위치) 내림차순 y sort
                    last_midrid_coordinates.sort(key=lambda x: x[1], reverse=True)
                else:
                    last_midrid_coordinates.sort(key=lambda x: x[1])
            else:
                if x_center_last < nearest_point_of_last_midrid[0]:
                    # midrid에 가장 가까운 point가 last midrid의 ceneter x coordinate보다 우측에 있으면 내림차순 x sort
                    last_midrid_coordinates.sort(reverse=True)
                else:
                    last_midrid_coordinates.sort()

            # last_midrid_coordinates를 3분할 후, 각각 y 또는 x sort를 진행한다.
            last_midrid_coordinates = rearange_last_midrid_coordinates(last_midrid_coordinates)

            ### phase_3
            # combine last_midrid_coordinates and point_list_phase_2 to point_list_phase_3
            point_list_phase_3 = combind_two_list(point_list_phase_2, last_midrid_coordinates,
                                                  last_midrid_coordinates[-1])
            first_point_of_phase_2 = [point_list_phase_2[0]]
            point_list_phase_3 = combind_two_list(first_point_of_phase_2, point_list_phase_3, point_list_phase_3[-1])

            ### phase_4
            point_list_phase_4 = get_midpoint_in_three_point(point_list_phase_3, 12, first_point=True, last_point=True)
            point_list_phase_4 = get_midpoint_between_two_point_2(point_list_phase_4, 10, first_point=True,
                                                                       last_point=True)
            tomato_leaf_info["midrid_point_coordinate"] = point_list_phase_4
            tomato_leaf_info["center"] = point_list_phase_4[len(point_list_phase_4) // 2]

            point_dict_list.append(tomato_leaf_info)

        return point_dict_list

    def get_stem_info(self):
        stem_point_dict_list = []

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

            if "width" not in stem_info.keys() or \
                    "center" not in stem_info.keys():
                continue
            if "height" not in stem_info.keys():
                continue
            stem_point_dict_list.append(stem_info)

        return stem_point_dict_list

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
                    slope_top = (y_left_top - y_right_top) / (x_top - x_bottom)
                    slope_bottom = (y_left_bottom - y_right_bottom) / (x_top - x_bottom)
                
                if slope_bottom * slope_top > 0:
                    slope = (slope_bottom + slope_top) / 2  # stem이 완전한 수평이 아닌, 기울어진 수평일 경우
                else:
                    slope = 0  # stem이 완전한 수평인 경우

            if abs(slope) <= 0.1:
                inverse_slope = 10
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
                    if int(inverse_slope * x + inverse_alpha) - self.margin_error <= int(y) <= int(
                            inverse_slope * x + inverse_alpha) + self.margin_error:
                        edge_list.append([x, y])

            if len(edge_list) == 2:
                _width_coordinates = edge_list

            ## 일차함수를 지나는 점의 개수가 두개 이상이면
            ## 두 점들 중 같은 일차함수와 같은 기울기를 갖는 두 점을 탐색
            elif len(edge_list) > 2:
                flag = False
                for i in range(len(edge_list) - 1):
                    x_1, y_1 = edge_list[i]
                    for j in range(i+1, len(edge_list)):
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

                stem_coordinates_top = stem_coordinates_sorted[
                                              :int(len(stem_coordinates_sorted) / 5)]   # stem의 상단 또는 좌측
                stem_coordinates_bottom = stem_coordinates_sorted[
                    int((len(stem_coordinates_sorted) / 5) * 4):]   # stem의 하단 또는 우측
                if abs(inverse_slope) <= 0.1 :
                    edge_list_top =  edge_list_top = sorted(stem_coordinates_top, key=lambda x:(x[1], abs(x_center - x[0])))
                    edge_list_bottom = sorted(stem_coordinates_bottom, key=lambda x:(-x[1], abs(x_center - x[0])))

                ### height가 수평인 경우 x값이 가장 크거나 작고, y값이 center의 y값과 가장 가까운 두 점을 height으로 설정
                elif abs(inverse_slope) >= 10:
                    edge_list_top = sorted(stem_coordinates_top, key=lambda x:(x[0], abs(y_center - x[1])))
                    edge_list_bottom = sorted(stem_coordinates_bottom, key=lambda x:(-x[0], abs(y_center - x[1])))
                else:
                    ### stem의 상단 또는 좌측에서 center point와 가장 멀리 떨어져 있는 점
                    edge_list_top = sorted(stem_coordinates_top, key=lambda x:get_length(stem_info["center"], x), reverse= True)
                    ### stem의 하단 또는 우측에서 center point와 가장 멀리 떨어져 있는 점
                    edge_list_bottom = sorted(stem_coordinates_bottom, key=lambda x:get_length(stem_info["center"], x), reverse= True)


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
                fruit_only_dict = self.get_only_fruit_points_minimum_box_algorithm(outer_idx)
                if fruit_only_dict is not None:
                    fruit_point_dict["fruit_only"].append(fruit_only_dict)
            elif len(inner_info_list) >= 1:  # cap또는 cap_2가 1개 이상 포함된 경우
                print("detected cap, fruit idx : {0}".format(outer_idx))
                fruit_point_dict = self.get_cap_fruit_points(outer_idx, inner_info_list, fruit_point_dict)
        return fruit_point_dict

    def get_only_fruit_points_minimum_box_algorithm(self, idx):
        fruit_segmentation_list = self.polygons[idx].copy()
        bbox_fruit = self.boxes[idx]

        fruit_segmentation_dict = dict(segmentation=fruit_segmentation_list.copy(),
                               type=self.plant_type,
                               bbox=bbox_fruit)

        fruit_only_dict = self.detect_minimum_box(fruit_segmentation_list, fruit_segmentation_dict)
        
        return fruit_only_dict

    def get_only_fruit_points(self, idx):
        fruit_points = self.polygons[idx].copy()
        fruit_points_for_search = self.polygons[idx].copy()
        bbox_fruit = self.boxes[idx]

        fruit_only_info = dict(segmentation=fruit_points.copy(),
                               type=self.plant_type,
                               bbox=bbox_fruit)

        fruit_only_info = self.common_fruit_only(fruit_points, fruit_points_for_search, fruit_only_info)

        return fruit_only_info

    def detect_minimum_box(self, segmentation, segmentation_dict):
        seg_x = []
        seg_y = []

        segmentation_list = segmentation
        
        for i in range(len(segmentation_list)):
            seg_point = segmentation_list[i]
            seg_x.append(seg_point[0])
            seg_y.append(seg_point[1])
            
        seg_nparray = np.asarray([np.asarray(i) for i in segmentation_list])
        
        center_x, center_y = np.mean(seg_x), np.mean(seg_y)
        
        rect = cv2.minAreaRect(seg_nparray)
        box = cv2.boxPoints(rect)
        box = box.astype(np.int32)
        
        pt_1, pt_2, pt_3 = box[0], box[1], box[3]   # min_x값을 기준으로 시계방향으로 point생성
        slope_h, _ = get_slope_alpha(pt_1[0], pt_1[1], pt_2[0], pt_2[1])
        slope_w, _ = get_slope_alpha(pt_1[0], pt_1[1], pt_3[0], pt_3[1])
        
        if abs(slope_h) < abs(slope_w):
            slope_h, slope_w = slope_w, slope_h     # 둘 중 기울기의 절대값이 더 큰 값을 height의 기울기로 설정
        

        h_p1, h_p2, height = self.find_width_or_heigth(slope_h, seg_nparray, center_x, center_y)
        
        w_p1, w_p2, width = self.find_width_or_heigth(slope_w, seg_nparray, center_x, center_y)
        
        segmentation_dict["height"] = [h_p1, h_p2]
        segmentation_dict["width"] = [w_p1, w_p2]
        segmentation_dict["center"] = [round(center_x), round(center_y)]
        
        return segmentation_dict
    
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
    def find_width_or_heigth(self, slope, seg_points, center_x, center_y):
        """
            기울기가 주어지면 해당 기울기를 지나는 가장 긴 길이의 point와 길이를 반환하는 함수
        """

        height = -1
        height_p1, height_p2 = None, None

        if slope!=100:      # slope가 수직이 아닌 경우
            for x,y in seg_points:      # contour의 모든 점들에 대해
                # slope_h의 기울기를 가지며 (x,y)를 지나는 일차함수를 구함
                alpha = y - slope * x

                for x2, y2 in seg_points:
                    if x != x2 and y!=y2:
                        if slope*x2 + alpha - 1 <= int(y2) <= slope*x2 + alpha + 1:     #오차범위 : 2
                            tmp_height = get_length((x,y), (x2,y2))
                            if tmp_height > height:
                                height = tmp_height
                                height_p1, height_p2 = [int(x),int(y)], [int(x2),int(y2)]
        
        # slope가 수직인 경우
        else:
            for x, y in seg_points:
                for x2, y2 in seg_points:
                    if y!= y2 and x==x2:
                        tmp_height = abs(y2-y)
                        if tmp_height > height:
                            height = tmp_height
                            height_p1, height_p2 = [int(x),int(y)], [int(x2),int(y2)]

        if not height_p1 or not height_p2:   # height를 찾지 못한 경우, 중심점을 지나는 일차함수를 사용
            alpha = center_y - slope * center_x
            tmp = []
            for x,y in seg_points:
                if slope * x + alpha - 3 <= int(y) <= slope * x + alpha + 3:
                    tmp.append((x,y))
            if len(tmp) >= 2:
                for i in range(len(tmp)):
                    _p1 = tmp[i]
                    for _p2 in tmp[i+1:]:
                        tmp_height = get_length(_p1, _p2)
                        if tmp_height > height:
                            height = tmp_height
                            height_p1, height_p2 = [int(_p1[0]),int(_p1[1])],[int(_p2[0]),int(_p2[1])]
        
        return height_p1, height_p2, height

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

            print(f'Tomato with cap detected: {outer_idx} with cap index {real_cap_idx}')
        elif real_cap_idx is not None and real_cap_2_idx is not None:
            # cap과 cap_2가 둘 다 있는 경우

            fruit_point_dict = self.get_cap_fruit_points_(outer_idx, [real_cap_idx, real_cap_2_idx], fruit_point_dict,
                                                          is_cap=True, is_cap_2=True)
            print(f'Tomato with cap detected: {outer_idx} with cap indexes {real_cap_idx} and {real_cap_2_idx}')
        elif real_cap_idx is None and real_cap_2_idx is None:
            # cap과 cap_2가 둘 다 없는 경우. 이런 경우는 code상 없지만, 만일을 위해 구현
            fruit_only_dict = self.get_only_fruit_points(outer_idx)
            if fruit_only_dict is not None:
                fruit_point_dict["fruit_only"].append(fruit_only_dict)

        elif real_cap_idx is None and real_cap_2_idx is not None:
            # cap은 없는데 cap_2가 있는 경우
            fruit_point_dict = self.get_cap_fruit_points_(outer_idx, real_cap_2_idx, fruit_point_dict,
                                                          is_cap=False, is_cap_2=True)

            print(f'Tomato with cap_2 only detected: {outer_idx} with cap_2 index {real_cap_2_idx}')
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
            fruit_info_dict = self.compute_fruit_w_h_common_fruit(outer_idx, inner_idx)
            if fruit_info_dict is not None:
                for key, item in fruit_info_dict.items():
                    fruit_info[key] = item

            if "height" not in list(fruit_info.keys()): return fruit_point_dict

            # width 계산
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

    def compute_fruit_w_h_common_fruit(self, outer_idx, inner_idx):
        # sub function
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
        def get_bottom_point(fruit_points, fruit_points_part_2, x_center_cap, y_center_cap):
            bottom_point = None
            # cap의 center point로부터 가장 먼 fruit point가 bottom point
            fruit_points_part_2 = fruit_points[int(len(fruit_points) * 2 / 3):]
            max_length = -1
            for point in fruit_points_part_2:  # cap의 center point로부터 가장 가까운 fruit point가 top point
                tmp_length = get_length([x_center_cap, y_center_cap], point)
                if max_length < tmp_length:
                    bottom_point = point
                    max_length = tmp_length
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
        bottom_point = get_bottom_point(fruit_points, fruit_points_part_2, x_center_cap, y_center_cap)

        if top_point is None or bottom_point is None: return None

        fruit_info["height"] = [top_point, bottom_point]
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
                        half_horizon_length = math.sqrt(
                            math.pow(width_center_point[0] - point[0], 2) + math.pow(width_center_point[1] - point[1],
                                                                                     2))
                        horizon_length = math.sqrt(
                            math.pow(width_point[0][0] - point[0], 2) + math.pow(width_point[0][1] - point[1], 2))

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

        if "width" not in fruit_info.keys() or \
                "height" not in fruit_info.keys(): return None
        return fruit_info
    

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
    

    def get_skeleton(self, mask):
        _, biimg = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
        # 거리변환
        dst = cv2.distanceTransform(biimg, cv2.DIST_L2, 5)

        # 거리 값을 0 ~ 255 사이로 정규화
        cv2.normalize(dst, dst, 0, 255, cv2.NORM_MINMAX)

        # 거리값에 threshold로 완전한 뼈대 찾기
        dst = dst.astype(np.uint8)
        skeleton = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, -2)
        return skeleton
        # 결과 출력
        # cv2.imshow("skel", skeleton)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

    def sort_midrib_points(self, polygons):
        
        polygons = polygons.copy()
        ordered = [polygons[0]]
        points = polygons[1:]

        while len(points) > 0:
            last_point = ordered[-1]
            distances = cdist([last_point], points)[0]
            nearest_index = np.argmin(distances)
            ordered.append(list(points[nearest_index]))
            points = np.delete(points, nearest_index, axis=0).tolist()
        
        return ordered


    # def sort_midrib_points(self, polygons):
        
    #     polygons = polygons.copy()
    #     start_point, end_point = self.find_first_point(polygons)
    #     print(start_point)
    #     ordered = [start_point]
    #     points = [x for x in polygons if not np.array_equal(x, start_point) and not np.array_equal(x, end_point)]

    #     while len(points) > 0:
    #         last_point = ordered[-1]
    #         distances = cdist([last_point], points)[0]
    #         nearest_index = np.argmin(distances)
    #         ordered.append(list(points[nearest_index]))
    #         points = np.delete(points, nearest_index, axis=0).tolist()
        
    #     ordered.append(end_point)
    #     return ordered
    
    # def find_first_point(self, polygons):
    #     # U자 모양을 가정하고 y좌표가 가장 큰 두점을 찾음
    #     sorted_polys = sorted(polygons, key=lambda x:x[1], reverse=True)
    #     points = sorted_polys[:2]

    #     # 두 점 중 x좌표가 작은 점을 시작 점으로 선택
    #     start_point = min(points, key=lambda x:x[0])
    #     last_point = max(points, key=lambda x:x[0])
    #     return start_point, last_point

    def find_first_point(self, points):
        points = points.copy() 
        min_x_index, _ = min(enumerate(points), key=lambda p: p[1][0])
        min_y_index, _ = min(enumerate(points), key=lambda p: p[1][1])
        max_x_index, _ = max(enumerate(points), key=lambda p: p[1][0])
        max_y_index, _ = max(enumerate(points), key=lambda p: p[1][1])

        min_x_path = self.nearest_neighbor(points, min_x_index)
        min_y_path = self.nearest_neighbor(points, min_y_index)
        max_x_path = self.nearest_neighbor(points, max_x_index)
        max_y_path = self.nearest_neighbor(points, max_y_index)

        min_x_length = self.path_distance(points, min_x_path)
        min_y_length = self.path_distance(points, min_y_path)
        max_x_length = self.path_distance(points, max_x_path)
        max_y_length = self.path_distance(points, max_y_path)

        length_list = [min_x_length, min_y_length, max_x_length, max_y_length]
        path_list = [min_x_path, min_y_path, max_x_path, max_y_path]
        min_idx = length_list.index(min(length_list))
        return path_list[min_idx]

        




    def nearest_neighbor(self, points, start_idx):
        n = len(points)
        path = [start_idx]
        unvisited = set(range(n)) - {start_idx}

        while unvisited:
            last = path[-1]
            next_point = min(unvisited, key=lambda x: np.linalg.norm(np.array(points[x]) - np.array(points[last])))
            path.append(next_point)
            unvisited.remove(next_point)
        return path
    
    def two_opt(self, points, path):
        imporved = True
        best_distance = self.path_distance(points, path)

        while imporved:
            imporved = False
            for i in range(1, len(path) - 2):
                for j in range(i+1, len(path)):
                    new_path = path[:i] + path[i:j][::-1] + path[j:]
                    new_distance = self.path_distance(points, new_path)
                    if new_distance < best_distance:
                        path = new_path
                        best_distance = new_distance
                        imporved = True
                        break
                if imporved:
                    break
        return path

    def path_distance(self, points, path):
        return sum(np.linalg.norm(np.array(points[path[i]]) - np.array(points[path[i-1]])) for i in range(1, len(path)))
    

    def extract_skeleton(self, mask):
       size = np.size(mask)
       skel = np.zeros(mask.shape, np.uint8)
       _, img = cv2.threshold(mask, 127, 255, 0)
       element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
       done = False

       while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True
        
        points = np.column_stack(np.where(skel > 0))
        # print("points" , len(points))
        # cv2.imshow("skel", skel)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        return skel

