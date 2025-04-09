# from calendar import c
# from re import T
# from turtle import color
# from re import T
# from turtle import color
from email.utils import collapse_rfc2231_value
from multiprocessing.connection import wait
from operator import le
import re
from tkinter import N
from turtle import Turtle, color
from typing import Tuple
import cv2
from cv2 import polarToCart
from cv2 import imshow
from cv2 import waitKey
import numpy as np
import sys, os, math
import utils as c_utl
from utils import (get_length, get_length_list, get_max_min_coordi, 
                   get_slope_alpha, create_check_flag, get_width_point)

import detectron2
from detectron2.utils.visualizer import GenericMask

# 해당 code에서 허용하는 작물 종류
VALID_PLANTS = ["strawberry", "paprika", "melon", "cucumber", "onion", "seeding_tomato", "chilipepper_seed", "chili", "cucumber_seed", "tomato"] # "tomato"

# 각 작물별 object
common_object = ["midrid", "leaf", "stem", "flower", "fruit", "cap"]
PLANTS_OBJECT_IDX = dict(
    strawberry= common_object+ ["y_fruit"],
    paprika= common_object+ ["cap_2"],
    melon= common_object+ ["petiole"],
    tomato= common_object+ ["mid_midrid", "last_midrid", "first_midrid", "side_midrid"],      
    cucumber= common_object+ ["fruit_top", "fruit_bottom"],      # TODO : cap       petiole 없음
    onion= common_object+ ["stem_leaf", "first_leaf", "first_leaf_list", "stem_leaf_list"],     
    chilipepper_seed = common_object,
    chili = common_object,
    cucumber_seed = common_object,
    
    seeding_tomato = common_object,     # TODO : delete
    
    seedling_boar= ["seed", "sbox"]     # TODO: delete
)

# 개수를 count하는 object
COUNT_OBJECT =  ['flower', 'y_fruit']

# 서로 대응관계가 있는 object인지 확인해야 하는 object (외부, 내부 object는 서로 대응관계가 있다.)
# 외부 object       # 예시: (과일fruit== 외부),(꼭지cap== 내부)
common_outer = ["leaf", "fruit"]
OUTER_OBJECTS = dict(
    paprika = common_outer,
    strawberry = common_outer,
    melon = common_outer,
    cucumber = common_outer,
    tomato = ['midrid', 'fruit'],
    onion = ["fruit", "stem_leaf"],
    chilipepper_seed = ['leaf'],
    chili = ['leaf'],
    cucumber_seed = ['leaf']
)

common_inner = ['midrid', 'cap']
# 내부 object 
INNER_OBJECTS = dict(
    paprika = common_inner + ['cap_2'],
    strawberry = common_inner,
    melon = common_inner,
    cucumber = common_inner,
    tomato = [ 'first_midrid', 'mid_midrid', 'last_midrid', 'cap'],
    onion = ['first_leaf', 'cap'],
    chilipepper_seed = ['midrid'],
    chili = ['midrid'],
    cucumber_seed = ['midrid']
)


# 최종적으로 계산된 좌표(장, 폭, center point 등)를 얻고자 하는 object
common_result = ['leaf', 'fruit', 'stem', 'flower']
RESULT_OBJECTS = dict(
    paprika = common_result,
    strawberry = common_outer + ['y_fruit'],
    melon = common_outer + ['petiole'],
    cucumber = common_outer,
    tomato =  ['midrid', 'stem', 'flower'],
    onion = ["fruit", "leaf"],
    chilipepper_seed = ['leaf', "stem"],
    chili = ['leaf', "stem"],
    cucumber_seed = ['leaf', "stem"]
    
)

# 좌표계산과는 관련 없이 가장자리 좌표(segmentation)를 저장할 object
SEG_OBJECT_LIST = [ "leaf", "fruit", "stem", "flower", "y_fruit", "petiole"]

# main function: calculate_coordinates
class Find_coordinates():
    def __init__(self, config, plant_type, class_name_list):
        self.config = config
        self.plant_type = plant_type
        self.class_name_list = class_name_list
        self.resize_scale = self.config.RESIZE_SCALE      # image가 너무 크면 resize_num 만큼 나눈다.
        
        self.margin_error = self.config.MARGIN_ERROR      # 오차범위

        self.centerbox_ratio = self.config.FRUIT_CENTER_BOX_RATIO       # fruit에서 cap이 fruit의 중앙의 얼마만큼의 범위 내에 속하는지를 결정하는 값
              
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
        self.outer_inner_idx = dict() 
        # outer_inner_idx[object_name].keys(): [outer_idx_1, outer_idx_2, ..., outer_idx_n]
        # outer_inner_idx[object_name][outer_idx_1]: [inner_idx_1, inner_idx_2, ... inner_idx_n]
        for outer_objects in OUTER_OBJECTS[self.plant_type]:    # 외부 object를 for문으로 나열
            self.outer_inner_idx[outer_objects] = dict()
            for outer_info in self.bbox[outer_objects]:         # 외부 object의 index를 나열
                bbox, outer_idx = outer_info
                self.outer_inner_idx[outer_objects][f'{outer_idx}'] = list()                 
                
                x_min, y_min, x_max, y_max = bbox
                width = (x_max - x_min)/2
                height = (y_max - y_min)/2
                
                # x_min, x_max, y_min, y_max : outer object의 bbox영역
                add_area_x_min = 0
                add_area_y_min = 0
                add_area_x_max = 0
                add_area_y_max = 0
                if outer_objects == "midrid" and self.plant_type != 'tomato':       
                    # 포함 area를 각 width, height의 1/5만큼 씩 잘라낸다. (정확한 midrid를 가려내기 위해)
                    add_area_x_min = -width/5
                    add_area_y_min = -height/5
                    add_area_x_max = -width/5
                    add_area_y_max = -height/5
                elif outer_objects == "fruit":
                    # 포함 area를 height의 1/4만큼 올린다. (cap은 보통 위에 위치)
                    # image는 위가 y값의 시작이다
                    add_area_y_min = +height/4
                    add_area_y_max = -height/4
                
                for inner_objects in INNER_OBJECTS[self.plant_type]:        # 내부 object를 나열
                    for inner_info in self.ceter_point[inner_objects]:      # 각 내부 object의 중앙점을 나열
                        center_point, inner_idx = inner_info
                        x_center, y_center = center_point
                        
                        # 내부 object의 중앙 점이 외부 object의 영역 안에 위치하는지 확인
                        # 위치하는 경우 외부-내부 object간 대응된다고 판단
                        if x_min - add_area_x_min < x_center and\
                        y_min - add_area_y_min < y_center and\
                        x_max + add_area_x_max > x_center and\
                        y_max + add_area_y_max > y_center:
                            if outer_idx not in self.useful_mask_idx:
                                self.useful_mask_idx.append(outer_idx)
                            self.useful_mask_idx.append(inner_idx)

                            # 서로 대응되는 object의 index를 key값 또는 list의 요소로 저장
                            self.outer_inner_idx[outer_objects][f'{outer_idx}'].append([inner_idx, inner_objects])
                                                # else:
                        #     self.outer_inner_idx[outer_objects][f'{outer_idx}'].append([None, None])
                                                            
       
    def set_dict(self):
        self.segmentations = dict()
        
        # 가장자리 points를 저장할 dict
        for object_name in SEG_OBJECT_LIST:
            self.segmentations[object_name] = []
        
        # index를 저장할 dict
        self.idx_dict = dict()
        for plant in VALID_PLANTS:
            for object_name in PLANTS_OBJECT_IDX[plant]:
                self.idx_dict[object_name] = []   
        
        # self.set_outer_inner_idx 에서 사용됨
        self.ceter_point = dict()   # 중앙점을 저장할 dict
        for object_name in INNER_OBJECTS[self.plant_type]:
            self.ceter_point[object_name] = []
        self.bbox = dict()          # bbox를 저장할 dict
        for object_name in OUTER_OBJECTS[self.plant_type]:
            self.bbox[object_name] = []
        
        # object의 개수를 저장할 dict
        self.count_object = dict()
        for obj_name in COUNT_OBJECT:
            self.count_object[obj_name] = 0
        
        # 좌표 계산에 사용할 mask의 idx만 list로 저장
        self.useful_mask_idx = []    
        
 

    def find_coordinate_slicing(self, continue_check, x, y, slope, alpha, edge_list, continue_num, skip_num, using_margin_error = True, margin_error = 3):
        if using_margin_error:
            margin_error = self.margin_error
        
        if continue_check:
            # int(y) == int(slope*x + alpha) 만으로는 1차함수 값에 대응되는 x, y값이 없는 경우가 있다.
            # self.margin_error 를 사용해서 오차범위를 조절해가며 1차함수 값에 대응되는 x, y값을 찾아본다.
            if (int(y) >= int(slope*x + alpha) - margin_error and int(y) <= int(slope*x + alpha) + margin_error) or\
               (int(y) >= int(slope*(x-margin_error) + alpha) and int(y) <= int(slope*(x+margin_error) + alpha)):
  
                edge_list.append([x, y])
                continue_check = False
        else :
            if continue_num == 0:
                continue_num = skip_num
                continue_check = True
            else:
                continue_num -=1
        
        return edge_list, continue_num, continue_check

    
    def select_points(self,points, width_or_height, dr):
        sorted_center_points = self.get_center_coordinates(points, width_or_height)
        sorted_selected_points = self.select_valiable_point(sorted_center_points, dr[0][0], dr[0][1])   
        sorted_selected_points = self.select_valiable_point(sorted_selected_points, dr[1][0], dr[1][1])   
        sorted_selected_points = self.select_valiable_point(sorted_selected_points, dr[2][0], dr[2][1])
        return sorted_selected_points
    
    def check_outer_exist(self, object_name):
        # if object_name not in list(self.outer_inner_idx.keys()):
        #     raise KeyError(f"{object_name} not in self.outer_inner_idx "
        #                    f"plant type is {self.plant_type}")
        
        if self.outer_inner_idx.get(object_name, None) is None: return False 
        if len(list(self.outer_inner_idx[object_name].keys())) !=0: return True
        else: return False
                
                
    def compute_width_height(self, bbox):
        x_min, y_min, x_max, y_max = bbox
        return x_max-x_min, y_max-y_min  
    
    def return_width_or_height(self, bbox):
        width, height = self.compute_width_height(bbox)
        more_longer = "width" if width > height else "height"
        return more_longer
        
        
    def get_object_idx(self, object_name):
        idx_list = []
        for idx in self.idx_dict[object_name]:
            if idx not in self.useful_mask_idx: continue
            idx_list.append(idx)
        
        return idx_list
        
        
    def compute_center_point(self, bbox):
        x_min, y_min, x_max, y_max = bbox
        return [int((x_min+x_max)/2), int((y_min+y_max)/2)]
    
                
    def contain_idx2dict(self, outputs):
        """
            self.idx_dict에 각각의 class name에 해당하는 key를 정의하고
            해당 class name에 부합하는 idx를 key값에 list로 할당
        """        
        # leaf에 포함되는 midrid를 찾기 위해 leaf와 midrid를 구분
        for idx, output in enumerate(outputs["instances"].pred_classes):
            
            for object_name in PLANTS_OBJECT_IDX[self.plant_type]:
                if self.class_name_list[output] == object_name:
                    print(f'Detected part: {idx} - {object_name}')  
                    self.idx_dict[object_name].append(idx)     # 해당 idx를 저장
                    
                    if object_name in INNER_OBJECTS[self.plant_type]:
                        # inner object는 서로 대응관계가 있는 object인지 확인하기 전 까진 self.useful_mask_idx에 append하지 않는다.
                        self.ceter_point[object_name].append([self.compute_center_point(self.boxes[idx]), idx])
                    elif object_name in OUTER_OBJECTS[self.plant_type]:
                        self.bbox[object_name].append([self.boxes[idx], idx])
                        if object_name == "fruit":      # fruit는 대응관계인 cap이 없어도 좌표찾기를 적용할 것이기 때문에 append
                            self.useful_mask_idx.append(idx)    
                            # leaf인 경우는 midrid와 대응되지 않는 경우 좌표찾기 적용하지 않을 예정
                    else:
                        self.useful_mask_idx.append(idx)
                    
                    if object_name in COUNT_OBJECT:
                        self.count_object[object_name] += 1


    def set_boxes_polygons(self, height, width, predictions):
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        
        
        self.boxes = list()
        for box in boxes.tensor.numpy():
            self.boxes.append([int(box[0]), int(box[1]), int(box[2]), int(box[3])])

        masks = np.asarray(predictions.pred_masks)
        masks = [GenericMask(x, height, width) for x in masks]
       
        # 모든 polygon(가장자리 points)를 list에 저장
        self.polygons = list()
        for mask in masks: 
            points = list()
            for segment in mask.polygons: 
                # segment : coordinates of segmentation boundary
                x_coordinates = segment.reshape(-1, 2)[:, 0]    # segmentation에서 x좌표만 extract
                y_coordinates = segment.reshape(-1, 2)[:, 1]    # segmentation에서 y좌표만 extract

                x_coordinates, y_coordinates = (list(map(int, x_coordinates/self.resize_scale)),   
                                                list(map(int, y_coordinates/self.resize_scale)))       # adjust coordinates for resized image
                            
                for x, y in zip(x_coordinates, y_coordinates):
                    points.append([x, y])
   
            self.polygons.append(points)
            
    
    def set_segmentations_dict(self, idx):
        for object_name in RESULT_OBJECTS[self.plant_type]:
            if object_name not in SEG_OBJECT_LIST:
                continue
            if idx in self.idx_dict[object_name]:
                if object_name in COUNT_OBJECT:
                    x_min, y_min, x_max, y_max = self.boxes[idx]
                    self.segmentations[object_name].append([int(x_min), int(y_min), int(x_max), int(y_max)])
                else:
                    self.segmentations[object_name].append(self.polygons[idx].copy())
                
    
    # main function       
    def calculate_coordinates(self, outputs, img ):
        if self.plant_type not in VALID_PLANTS: return None, None, None, None
        predictions = outputs["instances"].to("cpu")
        if not predictions.has("pred_masks"):
            # mask가 한 개도 없으면 save file을 하지 않음
            has_mask = False
            return img, has_mask, None, None
        height = img.shape[0]
        width = img.shape[1]
        self.img = cv2.resize(img, (int(width/self.resize_scale), int(height/self.resize_scale)))
        
        # 모든 polygon(가장자리 points)를 list에 저장
        self.set_boxes_polygons(height, width, predictions)     # make self.boxes, self.polygons
        self.contain_idx2dict(outputs)                          # 모든 object의 index를 dict에 저장 
        self.set_outer_inner_idx()                              # 서로 대응되는 object의 index를 dict의 key값과 그 안의 list의 요소로 저장                 
        
        for idx, _ in enumerate(self.boxes):
            if len(self.polygons[idx]) == 0: continue       # object가 detecting은 됐지만 points(polygon)가 없는 경우도 있다.
            self.set_segmentations_dict(idx)                # self.segmentations에 각 object name별로 index저장
        
        # 여기서부터 좌표 계산
        coordinates_dict = {}     # 각 계산한 좌표를 저장할 dict
        if len(self.get_object_idx("stem")) !=0:
            if self.plant_type in ["seeding_tomato", "chilipepper_seed", "chili", "cucumber_seed"]:     # 육묘인 경우 stem은 '초 장'을 의미
                stem_point_list = self.get_stem_seed_info()
            else:
                stem_point_list= self.get_stem_info()
            coordinates_dict["stem"] = stem_point_list

        if len(self.get_object_idx("leaf")) !=0:
            if self.plant_type == "onion":  
                # onion은 다른 작물의 leaf계산과는 다른 방법으로 좌표를 계산한다
                coordinates_dict["leaf"] = self.get_onion_leaf_info()         
        
        # self.check_outer_exist("leaf") : leaf가 외부 object인 작물인 경우
        # 해당되는 작물들: paprika, melon, cucumber, strawberry, chili, chilipepper_seed, cucumber_seed
        if self.check_outer_exist("leaf"): 
            coordinates_dict["leaf"] = self.get_draw_leaf_info() 
        
        # self.check_outer_exist("midrid") : midrid가 외부 object인 작물인 경우. 이는 tomato밖에 없다
        if self.check_outer_exist("midrid"):        # run only tomato 
            coordinates_dict["leaf"] = self.get_tomato_leaf_coordinates()

        # self.check_outer_exist("stem_leaf") : stem_leaf가 외부 object인 작물인 경우. 이는 onion밖에 없다
        if self.check_outer_exist("stem_leaf"):        # run only onion 
            # onion의 leaf 개수 구하는 code
            coordinates_dict["onion_leaf_num"] = self.count_onion_leaf_num()     
        
        # self.check_outer_exist("petiole") : petiole가 외가 object인 작물인 경우. 이는 melon밖에 없다             
        if len(self.get_object_idx("petiole")) !=0:       # run only melon
            # petiole_meta_list : [idx, petiole_mask_coordinates, box_width, box_height]
            coordinates_dict["petiole"] = self.get_petiole_info()

        if self.check_outer_exist("fruit"):
            fruit_point_dict = self.get_draw_fruit_info()
            coordinates_dict["fruit"] = fruit_point_dict
        
        # object counting 
        for object_name in COUNT_OBJECT:
            bbox_list = list()
            for object_idx in self.idx_dict[object_name]:
                bbox_list.append(self.boxes[object_idx])
            
            if self.count_object[object_name] ==0: continue
            coordinates_dict[object_name] = dict(count = self.count_object[object_name],        # 1개의 image에 detecting된 object_name의 개수
                                                 bbox = bbox_list)
            
        has_mask = True
     

        self.pop_none_object()
        return img, has_mask, coordinates_dict, self.segmentations
    


##
    def get_tomato_center_points(self, 
                                 points, 
                                 object, 
                                 width_or_height,
                                 mid_midrid_center_points = None, 
                                 img = None):

        if object == "midrid":
            dr = [[2, 2], [3, 1], [100, 3]]
            selected_point_x_sort = self.select_points(points, "width", dr)
            selected_point_y_sort = self.select_points(points, "height", dr)
            
            # selected_point_x_sort와 selected_point_y_sort를 사용하여 2단계의 parsing을 통해 최종 result_points를 계산 
            point_list_phase_1 = []
            for selected_points in [selected_point_x_sort, selected_point_y_sort]:
                for point in selected_points:
                    point_list_phase_1.append(point)
                    
            if mid_midrid_center_points is not None:
                for point in mid_midrid_center_points:
                    point_list_phase_1.append([int(point[0]), int(point[1])])      
            
            point_list_phase_2 = []
            before_point = None
            for i in range(len(point_list_phase_1)):
                if i == 0: 
                    point_list_phase_2.append(point_list_phase_1[0])
                    before_point = point_list_phase_1[0]
                min_length = 10000
                next_point = None
                for point_j in point_list_phase_1:
                    if point_j in point_list_phase_2: continue    
                    length =  get_length(before_point, point_j)
                    if min_length>length:
                        min_length = length
                        next_point = point_j
                if next_point is not None:
                    point_list_phase_2.append(next_point)
                    before_point = next_point
            
            result_point = point_list_phase_2   


        elif object == "first":
            sorted_center_coordinates = self.get_center_coordinates(points, width_or_height)
            
            if width_or_height == "height": sorted_center_coordinates.sort(key = lambda x:x[1])
            else: sorted_center_coordinates.sort()
            result_point = self.select_valiable_point(sorted_center_coordinates, 100, 10)       # 9/10 만큼의 point를 제거
                

        elif object[0] == "last":
            dr = [[1.5, 2], [3, 1], [100, 3]]
            selected_point_x_sort = self.select_points(points, "width", dr)
            selected_point_y_sort = self.select_points(points, "height", dr)

            point_list_phase_1 = []
            for parsed_point in [selected_point_x_sort, selected_point_y_sort]:
                for point in parsed_point:
                    point_list_phase_1.append(point)

            point_list_phase_2 = []
            before_point = None
            for i in range(len(point_list_phase_1)):
                if i == 0: 
                    point_list_phase_2.append(point_list_phase_1[0])
                    before_point = point_list_phase_1[0]

                min_length = 10000
                next_point = None
                
                for point_j in point_list_phase_1:
                    if point_j in point_list_phase_2: continue    
                    length = get_length(before_point, point_j)
                    if min_length>length:
                        min_length = length
                        next_point = point_j

                if next_point is not None:
                    point_list_phase_2.append(next_point)
                    before_point = next_point

            result_point = point_list_phase_2    
         
        return result_point


    def append_side_midrid_idx_to_total_midrid_idx_list(self, idx, side_midrid_mask_coordinates, total_midrid_idx_list):
        x_min_side, y_min_side, x_max_side, y_max_side  = self.boxes[idx]
        side_width, side_height = x_max_side - x_min_side, y_max_side - y_min_side
        if side_width < side_height :   # 세로로 긴 경우    y좌표 sort
            side_midrid_mask_coordinates.sort(key = lambda x:x[1])
        else:                           # 가로로 긴 경우    x좌표 sort
            side_midrid_mask_coordinates.sort()

        # total_midrid_idx_list[0] : [midrid_idx, first_midrid_idx, mid_midrid_idx_list, last_midrid_idx]
        for leaf_idx, total_midrid_idx in enumerate(total_midrid_idx_list):
            mid_midrid_idx_list = total_midrid_idx[2]
            break_check_flag_1 = False
            for mid_midrid_idx in mid_midrid_idx_list:
                break_check_flag_2 = False

                # side_midrid_mask 를 10개로 분할
                side_part_bbox_list = []
                for i in range(10):
                    x_min, y_min, x_max, y_max = 10000, 10000, -1, -1
                    side_midrid_mask_part = side_midrid_mask_coordinates[int(len(side_midrid_mask_coordinates)*(i/10)):int(len(side_midrid_mask_coordinates)*((i+1)/10))]
                    
                    for side_midrid_point in side_midrid_mask_part:
                        # 각 part마다 x_min, y_min, x_max, y_max를 계산
                        if side_midrid_point[0] < x_min:
                            x_min = side_midrid_point[0]
                        if side_midrid_point[1] < y_min:
                            y_min = side_midrid_point[1]
                        if side_midrid_point[0] > x_max:
                            x_max = side_midrid_point[0]
                        if side_midrid_point[1] > y_max:
                            y_max = side_midrid_point[1]
                    
                    # x_min_midrid, y_min_midrid, x_max_midrid, y_max_midrid = self.boxes[mid_midrid_idx]
                    side_part_bbox = x_min, y_min, x_max, y_max
                    side_part_bbox_list.append(side_part_bbox)
                    iou = c_utl.comfute_iou(side_part_bbox, self.boxes[mid_midrid_idx])     # mid_midrid와 side_midrid_part 사이의 IOU계산            
                    if iou == 0:
                        continue
                    else:
                        break_check_flag_2 = True
                        iou_side_part_bbox = side_part_bbox
                    

                if break_check_flag_2:
                    break_check_flag_1 = True
                    break
            
            if break_check_flag_1:
                # total_midrid_idx_list는 한 개의 image에 있는 leaf의 개수만큼의 len을 갖는다.
                # total_midrid_idx_list[leaf_idx] : [midrid_idx, first_midrid_idx, mid_midrid_idx_list, last_midrid_idx, side_midrid_meta_list]
                #   side_midrid_meta_list : [[idx, iou_side_part_bbox, side_part_bbox_list], ...]
                if len(total_midrid_idx_list[leaf_idx]) == 4:   # side_midrid_idx 가 한 개도 포함되지 않은 leaf인 경우
                    side_midrid_meta_list = []
                    side_midrid_meta_list.append([idx, iou_side_part_bbox, side_part_bbox_list])
                    total_midrid_idx_list[leaf_idx].append(side_midrid_meta_list)
                elif len(total_midrid_idx_list[leaf_idx]) == 5: # 한 개 이상의 side_midrid_idx가 이미 포함된 leaf인 경우
                    total_midrid_idx_list[leaf_idx][4].append([idx, iou_side_part_bbox, side_part_bbox_list])
                break
        
        return total_midrid_idx_list


    def get_stem_seed_info(self):       
        tmp_list = []
            
        for idx in self.get_object_idx("stem"):
            stem_coordinates = self.polygons[idx].copy()
            midrid_center_coordinates = self.get_sorted_center_points(self.polygons[idx].copy(),
                                                                      width_or_height = self.return_width_or_height(self.boxes[idx]))
            tmp_dict = {}
            tmp_dict["segmentation"] = stem_coordinates.copy()
            tmp_dict["type"] = self.plant_type
            tmp_dict["bbox"] = self.boxes[idx]
            # tmp_dict["width"]

            midrid_center_coordinates_part_1 = midrid_center_coordinates[:len(midrid_center_coordinates)//2]
            midrid_center_coordinates_part_2 = midrid_center_coordinates[len(midrid_center_coordinates)//2:]
            
            
            point_list = []
            num_spot = self.config.NUM_SPOT       # c_cfg.NUM_SPOT == 10 이면 10개의 point를 찍는다.
            num_spot_1 = int(num_spot/3)

            count_bot = 1
            ### select particular point
            for i, center_coordinate in enumerate(midrid_center_coordinates_part_1):
                if i  == int(len(midrid_center_coordinates_part_1)*(count_bot/num_spot_1)) or i == 0 or i == (len(midrid_center_coordinates_part_1) -1):    
                    count_bot +=1
                    point_list.append(center_coordinate)
            
            num_spot_2 = int(num_spot/5)

            count_bot = 1
            ### select particular point
            for i, center_coordinate in enumerate(midrid_center_coordinates_part_2):
                if i  == int(len(midrid_center_coordinates_part_2)*(count_bot/num_spot_2)) or i == 0 or i == (len(midrid_center_coordinates_part_2) -1):    
                    count_bot +=1
                    point_list.append(center_coordinate)
            
            tmp_dict["height"] = point_list
            

            midpoint_of_last_y_point = (point_list[-1][1] + point_list[-2][1])//2
            
            stem_coordinates.sort(key = lambda x:x[1])
            last_part_stem_of_lastpoint = stem_coordinates[int(len(stem_coordinates)*4/5):]
            
            for i in range(5):  # 오차
                min_x_point, max_x_point = 100000, -1
                
                for point in last_part_stem_of_lastpoint:
                    # print(f"midpoint_of_lastpoint : {midpoint_of_lastpoint},        point : {point}")
                    if midpoint_of_last_y_point == point[1]:    # midpoint_of_last_y_point와 같은 y값을 가진 좌표인 경우
                        if min_x_point > point[0]:              # 해당 x좌표가 줄기의 왼쪽 부분일 경우
                            min_x_point = point[0]
                        if max_x_point < point[0]:              # 해당 x좌표가 줄기의 오른쪽 부분일 경우
                            max_x_point = point[0]

                if min_x_point == max_x_point or max_x_point == -1:     # midpoint_of_last_y_point와 같은 y값을 가진 좌표가 두 개 미만으로 탐색된 경우
                    midpoint_of_last_y_point += 1   # midpoint_of_last_y_point의 y좌표 증가
                else:
                    tmp_dict["width"] = [[min_x_point, midpoint_of_last_y_point], [max_x_point, midpoint_of_last_y_point]]
                    break

            if "width" not in tmp_dict.keys() or "height" not in tmp_dict.keys(): continue

            tmp_list.append(tmp_dict)
            
        return tmp_list

    
    

# onion의 leaf개수 count
    def count_onion_leaf_num(self):
        onion_leaf_meta_list = []
        for outer_idx, inner_info_list in  self.outer_inner_idx['stem_leaf'].items():
            outer_idx = int(outer_idx)
            
            stem_leaf_center_point = self.compute_center_point(self.boxes[outer_idx])
      
            first_leaf_center_list = []  
            for inner_idx, _ in inner_info_list:
                first_leaf_center_list.append(self.compute_center_point(self.boxes[inner_idx]))
            
            tmp_dict = {'main_center_point' : stem_leaf_center_point,
                        'count_leaf' : len(first_leaf_center_list),
                        'leaf_center_points' : first_leaf_center_list}
            
            onion_leaf_meta_list.append(tmp_dict)
        
        return onion_leaf_meta_list


#### stem에 관련된 function
    def get_stem_info(self):
        stem_point_dict_list = []
        
        for idx in self.get_object_idx("stem"):
            stem_points = self.polygons[idx].copy()
            bbox_stem = self.boxes[idx].copy()
            
            stem_info = dict(segmentation = stem_points.copy(),
                             type = self.plant_type,
                             bbox = bbox_stem)
      
            
            # tmp_dict 의 key "width", "height", "center" 추가
            stem_info = self.get_stem_points(stem_points, 
                                            stem_info, 
                                            self.return_width_or_height(bbox_stem))

            
            
            if "width" not in stem_info.keys() or\
                "center" not in stem_info.keys(): continue
            
            if self.plant_type == "onion":        # onion일 경우 bbox를 stem의 하단 부근으로 다시 계산        
                x_min, y_min, x_max, y_max = bbox_stem
                x_point, _ = stem_info["center"]
                coordinates = stem_points
                if  x_max - x_min < y_max - y_min :     # stem이 수직으로 긴 경우
                    coordinates.sort(key = lambda x : x[1])
                    part_stem_coordinates = coordinates[len(coordinates)//2:]       # 절반만 자른다.
                else:                                   # stem이 수평으로 긴 경우
                    coordinates.sort()  # x좌표 sort
                    if stem_info["center"][0] < x_point : # 왼쪽 하단      
                        part_stem_coordinates = coordinates[:len(coordinates)//2]
                    else : # x좌표 우측 하단
                        part_stem_coordinates = coordinates[len(coordinates)//2:]  
                
                # bbox를 다시 계산 
                x_max, y_max, x_min, y_min = -1, -1, 100000, 100000
                for point in part_stem_coordinates: 
                    if point[0] > x_max : x_max = point[0]
                    elif point[0] < x_min : x_min = point[0]
                    if point[1] > y_max : y_max = point[1]
                    elif point[1] < y_min : y_min = point[1]            
                stem_info['bbox'] = [x_min, y_min, x_max, y_max]     # 절반 자른 stem의 bbox  

                
                if "width" not in stem_info.keys() or "center" not in stem_info.keys(): continue
                stem_point_dict_list.append(stem_info)
            else:
                if "height" not in stem_info.keys(): continue
                stem_point_dict_list.append(stem_info)     
        
        return stem_point_dict_list
        
      


    def get_stem_points(self, stem_points, stem_info, w_h):
        if w_h == "height":
            stem_points.sort(key=lambda x: x[1])
        else:
            stem_points.sort()
        
        stem_coordinates_sorted = stem_points
        # stem의 중단부분 coordinates       1/4 ~ 3/4
        stem_coordinates_mid = stem_coordinates_sorted[int(len(stem_coordinates_sorted)/4):int(len(stem_coordinates_sorted)/4)*3]
        # stem의 중단 영역의 top부분(또는 좌측 부분) coordinates    1/4 ~ 1.3/4
        stem_coordinates_mid_top = stem_coordinates_sorted[int(len(stem_coordinates_sorted)/4):int((len(stem_coordinates_sorted)/4)*1.3)]
        # stem의 중단 영역의 bottom부분(또는 우측 부분) coordinates     2.7/4 ~ 3/4
        stem_coordinates_mid_bottom = stem_coordinates_sorted[int((len(stem_coordinates_sorted)/4)*2.7):int(len(stem_coordinates_sorted)/4)*3]
        # stem의 중단부분 coordinates(넓은 영역)    1/5 ~ 4/5
        stem_coordinates_mid_wide = stem_coordinates_sorted[int(len(stem_coordinates_sorted)/5):int(len(stem_coordinates_sorted)/5)*4]

        x_min, y_min, x_max, y_max = stem_info["bbox"]
        x_center, y_center = [int((x_min + x_max)/2), int((y_min + y_max)/2)]
        box_width, box_height = x_max - x_min, y_max - y_min
            
        
        x_bottom_left = None
        x_top_right = None
        y_left_bottom = None
        y_right_top = None
        if w_h == "height": # stem이 수직 형태인 경우
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
                if count == 2: break
                if y_stem == y_bottom and x_stem !=x_bottom_right:
                    x_bottom_left = x_stem      # [x_bottom_left, y_bottom] : stem의 3/4지점 좌측 하단 point 
                    count +=1       
                    continue
                if y_stem == y_top and x_stem !=x_top_left:
                    x_top_right = x_stem        # [x_top_right, y_top] : stem의 1/4지점 우측 상단 point 
                    count +=1
                    continue

        else: # stem이 수평 형태인 경우
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
                if count == 2: break
                if x_stem == x_bottom and y_stem !=y_right_bottom:
                    y_left_bottom = y_stem      # [x_bottom, y_left_bottom] : stem의 3/4지점 좌측 하단 point 
                    count +=1       
                    continue
                if x_stem == x_top and y_stem !=y_left_top:
                    y_right_top = y_stem        # [x_top, y_left_top] : stem의 1/4지점 우측 상단 point 
                    count +=1
                    continue
                    
        
        if (x_bottom_left is not None and x_top_right is not None) or (y_right_top is not None and y_left_bottom is not None):     
               
            if w_h == "height":
                # slope_left : stem의 좌측 point의 기울기
                # slope_right : stem의 우측 point의 기울기
                if x_top_right - x_bottom_right == 0 or x_top_left - x_bottom_left == 0:   # 완전한 수직인 경우   
                    slope_left = 100
                    slope_right = 100
                else : 
                    slope_left = (y_top - y_bottom) / (x_top_left - x_bottom_left)
                    slope_right = (y_top - y_bottom) / (x_top_right - x_bottom_right)
                
                if slope_right * slope_left > 0 :    slope = (slope_left + slope_right) / 2        # 두 기울기가 같은 방향으로 기울어져 있을 경우     
                else :  slope = 100                                # 두 기울기가 서로 음-양 값을 가진 경우
                    
                    
            elif w_h == "width":
                if y_right_bottom - y_left_bottom == 0 or y_left_top - y_right_top == 0:    # 완전한 수평인 경우  
                    slope_top = 0
                    slope_bottom = 0
                else : 
                    slope_top = (x_top - x_bottom) / (y_left_top - y_right_top)
                    slope_bottom = (x_top - x_bottom) / (y_right_bottom - y_left_bottom)
                
                if slope_bottom * slope_top > 0 :   slope = (slope_bottom + slope_top) / 2    # stem이 완전한 수평이 아닌, 기울어진 수평일 경우 
                else :  slope = 0                             # stem이 완전한 수평인 경우

            if abs(slope) <= 0.1 :   inverse_slope = 10     # stem이 거의 수평인 경우
            elif abs(slope) >= 10 :   inverse_slope = 0.1    # stem이 거의 수직인 경우
            else: 
                inverse_slope = -1/(slope)
                alpha = -1*x_center*slope + y_center
                inverse_alpha = -1*x_center*inverse_slope + y_center
            
            edge_list = [] 
            width_coordinates = None
            for x, y in stem_coordinates_mid:
                if abs(inverse_slope) >= 10:
                    if x_center == x : edge_list.append([x, y])
                elif abs(inverse_slope) <= 0.1:
                    if y_center == y : edge_list.append([x, y])
                else : 
                    if int(y) >= int(inverse_slope*x + inverse_alpha) - 1 and int(y) <= int(inverse_slope*x + inverse_alpha) + 1:       
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
                        width_coordinates = edge_list
   
            if width_coordinates is not None :    
                stem_info["width"] = width_coordinates
                
                width_1, width_2 = width_coordinates[0], width_coordinates[1]       
                x_center_point, y_center_point = (width_1[0] + width_2[0])//2,(width_1[1] + width_2[1])//2
                stem_info["center"] = [x_center_point, y_center_point] 
                
                if self.plant_type != "onion" : # onion은 height를 계산 안함
                    stem_coordinates_top_bottom = stem_coordinates_sorted[:int(len(stem_coordinates_sorted)/5)]     # stem의 상단 또는 좌측 
                    for coordinate in stem_coordinates_sorted[int((len(stem_coordinates_sorted)/5)*4):]:            # stem의 하단 또는 우측  
                        stem_coordinates_top_bottom.append(coordinate)      # stem_coordinates_top_bottom : stem의 양 끝 부분들만 분리

                    for i in range(self.margin_error):
                        continue_check, continue_num, edge_list, skip_num = create_check_flag(stem_coordinates_top_bottom)
                        for x, y in stem_coordinates_top_bottom:
                            if abs(slope) <= 0.1 : 
                                if y_center == y : edge_list.append([x, y])
                            elif abs(slope) >= 10 : 
                                if x_center == x : edge_list.append([x, y])
                            else: edge_list, continue_num, continue_check = self.find_coordinate_slicing(continue_check, x, y, slope, alpha, edge_list, continue_num, skip_num, False, i)
                                
                            if len(edge_list) == 2:
                                x_1, y_1, x_2, y_2 = edge_list[0][0], edge_list[0][1], edge_list[1][0], edge_list[1][1]
                                length = get_length([x_1, y_1], [x_2, y_2])
                                
                                if (w_h == "h" and length < (box_height)/4) or (w_h == "w" and length < (box_width)/4):
                                    edge_list.pop(-1)
                                    continue
                                else: 
                                    stem_info["height"] = edge_list
                                    break
                            else: continue
                        if "height" not in stem_info.keys(): continue
                        else : 
                            # 2개의 point였던 height에 center point를 추가
                            tmp_list = stem_info["height"]
                            stem_info["height"] = [tmp_list[0], stem_info["center"], tmp_list[1]]
                            break

        return stem_info
    
    
    def get_cap_fruit_points(self, outer_idx, inner_info_list, fruit_point_dict):  
        # sub_function
        def select_cap_idx(cap_idxs):
            if len(cap_idxs) > 1:  # cap이 2개 이상 포함된 경우
                inner_idx = None
                min_length = 100000
                for idx in cap_idxs:
                    if idx is None:
                        continue
                    length = get_length(self.compute_center_point(self.boxes[outer_idx]), 
                                        self.compute_center_point(self.boxes[idx]))
                    if length < min_length:
                        min_length = length
                        inner_idx = idx
                return inner_idx
            elif len(cap_idxs) == 1: return cap_idxs[0]
            elif len(cap_idxs) == 0: return None
        
        
        # cap과 cap_2를 분리
        cap_idx, cap_2_idx = [], []
        for inner_info in inner_info_list:
            inner_idx, inner_objects_name = inner_info
            if 'cap_2' == inner_objects_name:
                cap_2_idx.append(inner_idx)
            else:
                cap_idx.append(inner_info[0])
            
        real_cap_idx = select_cap_idx(cap_idx)
        real_cap_2_idx = select_cap_idx(cap_2_idx)
        if real_cap_idx is not None and real_cap_2_idx is None:
            # cap은 있는데 cap_2가 없는 경우
            fruit_point_dict = self.get_cap_fruit_points_(outer_idx, real_cap_idx, fruit_point_dict,
                                                            is_cap = True, is_cap_2 = False)
    
        elif real_cap_idx is not None and real_cap_2_idx is not None:
            # cap과 cap_2가 둘 다 있는 경우
            
            fruit_point_dict = self.get_cap_fruit_points_(outer_idx, [real_cap_idx, real_cap_2_idx], fruit_point_dict,
                                                            is_cap = True, is_cap_2 = True)
        elif real_cap_idx is None and real_cap_2_idx is None:
            # cap과 cap_2가 둘 다 없는 경우. 이런 경우는 code상 없지만, 만일을 위해 구현
            fruit_only_dict = self.get_only_fruit_points(outer_idx)
            if fruit_only_dict is not None: 
                fruit_point_dict["fruit_only"].append(fruit_only_dict)
            
        elif real_cap_idx is None and real_cap_2_idx is not None:
            # cap은 없는데 cap_2가 있는 경우
            fruit_point_dict = self.get_cap_fruit_points_(outer_idx, real_cap_2_idx, fruit_point_dict,
                                                            is_cap = False, is_cap_2 = True) 
                    
        return fruit_point_dict


#### fruit와 cap에 관련된 function
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
        
        fruit_point_dict = dict(fruit_only = [],
                                cap_fruit_side = [],
                                cap_fruit_above = [])

      
        
        for outer_idx, inner_info_list in  self.outer_inner_idx['fruit'].items():
            outer_idx = int(outer_idx)
            # print(f"outer_idx : {outer_idx}")
            # print(f"fruit_points : {self.polygons[outer_idx]}")
            # img = self.img
            # for point in self.polygons[outer_idx]:
            #     cv2.circle(img, point, radius=2, color=(0, 255, 255), thickness=-1)
            
            # cv2.imshow("img", img)
            # while True:
            #     if cv2.waitKey() == 27: break
            
            if len(inner_info_list) == 0:   # cap(꼭지)가 포함되지 않는 경우
                fruit_only_dict = self.get_only_fruit_points(outer_idx)
                if fruit_only_dict is not None: 
                    fruit_point_dict["fruit_only"].append(fruit_only_dict)
            elif len(inner_info_list) >= 1: # cap또는 cap_2가 1개 이상 포함된 경우                
                fruit_point_dict = self.get_cap_fruit_points(outer_idx, inner_info_list, fruit_point_dict)
        return fruit_point_dict
            
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
        for x_y_coordinate in fruit_points:        # fruit_coordinates 를 x값이 낮은 위치부터 slicing
            if x_coordinate_slicing != x_y_coordinate[0]:
                fruit_sorted_coordinates_top.append([x_coordinate_slicing, y_coordinate_max])
                fruit_sorted_coordinates_bottom.append([x_coordinate_slicing, y_coordinate_min])
                x_coordinate_slicing, y_coordinate_max, y_coordinate_min = x_y_coordinate[0], x_y_coordinate[1], x_y_coordinate[1] 
            elif x_coordinate_slicing is None:  # slicing을 처음 시작할 때
                fruit_sorted_coordinates_top.append([x_coordinate_slicing, y_coordinate_max])
                fruit_sorted_coordinates_bottom.append([x_coordinate_slicing, y_coordinate_min])
                x_coordinate_slicing, y_coordinate_max, y_coordinate_min = x_y_coordinate[0], x_y_coordinate[1], x_y_coordinate[1]
            else : # 같은 x좌표가 여러개인 경우, 그 중 y가 가장 큰 좌표와 작은 좌표를 얻어낸다. (x좌표 중복 제거)
                if y_coordinate_max < x_y_coordinate[1]: y_coordinate_max = x_y_coordinate[1]
                if y_coordinate_min > x_y_coordinate[1]: y_coordinate_min = x_y_coordinate[1]


        # fruit_sorted_coordinates_top : fruit의 위쪽 coordinates
        # fruit_sorted_coordinates_bottom : fruit의 아래쪽 coordinates
        check_idx = 0
        length_fruit = -1
        tmp_idx = 0
        if len(fruit_sorted_coordinates_top) == len(fruit_sorted_coordinates_bottom):
            # 두 list의 len이 같다면 바로 length계산
            for top_coordi, bottom_coordi in zip(fruit_sorted_coordinates_top, fruit_sorted_coordinates_bottom):
                tmp_idx +=1
                # 가장 큰 length를 찾아 index를 보관 후 좌표를 얻는다.
                if length_fruit < top_coordi[1] - bottom_coordi[1]:
                    length_fruit = top_coordi[1] - bottom_coordi[1]
                    check_idx = tmp_idx
            x_1, y_1 = fruit_sorted_coordinates_top[check_idx]
            x_2, y_2 = fruit_sorted_coordinates_bottom[check_idx]
        else:
            # 두 list의 len이 같지 않다면 작은 len을 가진 list를 기준으로
            min_len = min(len(fruit_sorted_coordinates_top), len(fruit_sorted_coordinates_bottom))
            add_value_bottom = 0
            add_value_top = 0
            for i in range(min_len):
                if fruit_sorted_coordinates_top[i + add_value_top][0] == fruit_sorted_coordinates_bottom[i + add_value_bottom][0]:
                    tmp_idx +=1
                    if length_fruit < top_coordi[1] - bottom_coordi[1]:
                        length_fruit = top_coordi[1] - bottom_coordi[1]
                        check_idx = tmp_idx
                    # 높은 x값을 가진 coordinates list는 다음 iteration에서 add_value에 의한 -를 통해 index를 유지시켜준다.
                elif fruit_sorted_coordinates_top[i + add_value_top][0] < fruit_sorted_coordinates_bottom[i + add_value_bottom][0]:
                    add_value_bottom -=1
                elif fruit_sorted_coordinates_top[i + add_value_top][0] > fruit_sorted_coordinates_bottom[i + add_value_bottom][0]:
                    add_value_top -= 1
                    
        x_1, y_1 = fruit_sorted_coordinates_top[check_idx]
        x_2, y_2 = fruit_sorted_coordinates_bottom[check_idx]

        fruit_only_info["height"] = [[x_1, y_1], [x_2, y_2]]
    
        ## find width point
        slope, _ = get_slope_alpha(x_1, y_1, x_2, y_2)
        
        if slope == 0:
            inverse_slope = 20
        else: inverse_slope = (-1)/slope
        
        
        mid_point_alpha = []
        if self.plant_type in ["paprika", "strawberry", "onion", "tomato", "melon"]:
            mid_point_num = 5
            x_val, y_val = abs(x_1-x_2)/mid_point_num, abs(y_1-y_2)/mid_point_num 
            num_list = [i - mid_point_num//2 for i in range(mid_point_num)]     
            for i in num_list:
                x_mid, y_mid = (x_1+x_2)/2 + (x_val*i), (y_1+y_2)/2 + (y_val*i)
                alpha = y_mid - x_mid*(inverse_slope)
                mid_point_alpha.append([alpha, [int(x_mid), int(y_mid)]])
                
            
        width_length_list = []
        for mid_point in mid_point_alpha:
            alpha = mid_point[0]

            check_boolean = False
            for i in range(self.margin_error):
                continue_check, continue_num, edge_list, skip_num = create_check_flag(fruit_points_for_search)
                for x, y in fruit_points_for_search:
                    edge_list, continue_num, continue_check = self.find_coordinate_slicing(continue_check, x, y, inverse_slope, alpha, edge_list, continue_num, skip_num, False, i)
                    if len(edge_list) == 2:
                        width_coordinates = [edge_list[0], edge_list[1]]
                        check_boolean = True
                        break
                if not check_boolean: continue
                else : break

            if width_coordinates is None: continue
            length = get_length(width_coordinates[0], width_coordinates[1])
            width_length_list.append([length, width_coordinates, mid_point[1]])
        
        # 폭(width) 좌표를 구하지 못하는 경우 return None
        if len(width_length_list) == 0: return None
        
    
        max_length = -1
        for width_info in width_length_list:
            width_length, width_coordi, mid_center_point = width_info       
            if max_length<width_length:
                max_length = width_length
                
                fruit_only_info["width"] = width_coordi
                fruit_only_info["center"] = mid_center_point
                
        
        if "width" not in fruit_only_info.keys() or\
            "height" not in fruit_only_info.keys(): return None
    
        return fruit_only_info  
    
    def get_cucumber_mid_points(self, points, point_num, get_first):
        # get mid_points (mid_points == length of cucumber)
        mid_points_list_step_1 = []   # cucumber의 높이에 따른 x좌표의 중점을 저장할 list
        num_of_point = point_num   # cucumber의 length를 몇 개의 point를 나눌건지 결정하는 변수 
        point_count = int(len(points)/num_of_point)
        count_down = False

        keep_point = None
        exist_point = False
        for i, coordinate in enumerate(points):
            if get_first:
                if i == 0 :                                         # first point
                    mid_points_list_step_1.append(tuple(coordinate))
                    count_down = True
        
            if i == len(points) - 1 :                # last point 
                mid_points_list_step_1.append(tuple(coordinate))   

            if count_down : 
                point_count -=1
                if point_count == 0 : 
                    count_down = False
                    point_count = int(len(points)/num_of_point)          
                else : continue
                    
            if keep_point == None:  
                keep_point = coordinate
                exist_point = False
            else : 
                if exist_point : continue
                elif keep_point == coordinate : continue
                else : 
                    x_new, y_new = coordinate
                    x_b, y_b = keep_point
                    
                    center_fruit_point = (int((x_new + x_b)/2), int((y_new + y_b)/2)) 
                    mid_points_list_step_1.append(center_fruit_point)
                    exist_point = True
                    keep_point = None
                    count_down = True
        
        mid_points_list_step_2 = []
        for idx in range(len(mid_points_list_step_1)):
            if idx == 0 or idx == len(mid_points_list_step_1)-1 : 
                mid_points_list_step_2.append(mid_points_list_step_1[idx])
                continue
            x_before, y_before = mid_points_list_step_1[idx-1][0], mid_points_list_step_1[idx-1][1]
            x_this, y_this = mid_points_list_step_1[idx][0], mid_points_list_step_1[idx][1]
            x_next, y_next = mid_points_list_step_1[idx+1][0], mid_points_list_step_1[idx+1][1]
                                
            slope_before, _ = get_slope_alpha(x_before, y_before, x_this, y_this) 
            slope_next, _ = get_slope_alpha(x_this, y_this, x_next, y_next)
            if slope_before * slope_next < 0 : 
                x_this, y_this = int((x_before + x_next)/2), int((y_before + y_next)/2)
                mid_points_list_step_2.append([x_this, y_this])
            else: mid_points_list_step_2.append(mid_points_list_step_1[idx])
        # mid_points_list_step_2 = mid_points_list_step_1
        
        mid_points_list_step_3 = []
        for idx in range(len(mid_points_list_step_2)):
            if idx == 0 or idx == len(mid_points_list_step_2)-1 : 
                mid_points_list_step_3.append(mid_points_list_step_2[idx])
                continue
            x_before, y_before = mid_points_list_step_2[idx-1][0], mid_points_list_step_2[idx-1][1]
            x_this, y_this = mid_points_list_step_2[idx][0], mid_points_list_step_2[idx][1]
            x_next, y_next = mid_points_list_step_2[idx+1][0], mid_points_list_step_2[idx+1][1]   
            if (slope_before > 1 and 0 < slope_next < 1) or (slope_next > 1 and 0 < slope_before < 1) or\
                (slope_before < -1 and -1 < slope_next < 0) or (slope_next < -1 and -1 < slope_before < 0):
                    
                x_this, y_this = int((x_before + x_next)/2), int((y_before + y_next)/2)
                mid_points_list_step_3.append([x_this, y_this])
            else: mid_points_list_step_3.append(mid_points_list_step_2[idx])

        mid_points_list_step_4 = []
        for idx in range(len(mid_points_list_step_3)):
            if idx == len(mid_points_list_step_3)-1 : 
                mid_points_list_step_4.append(mid_points_list_step_3[idx])
                break
            mid_points_list_step_4.append(mid_points_list_step_3[idx])
            x_this, y_this = mid_points_list_step_3[idx][0], mid_points_list_step_3[idx][1]
            x_next, y_next = mid_points_list_step_3[idx+1][0], mid_points_list_step_3[idx+1][1]
            mid_points_list_step_4.append([int((x_this + x_next)/2), int((y_this + y_next)/2)])

        return mid_points_list_step_4
    
    
    def cucumber_fruit_cap(self, outer_idx, inner_idx):
        bbox_fruit = self.boxes[outer_idx]
        fruit_points = self.polygons[outer_idx].copy()
        fruit_points_for_search = fruit_points.copy()
        if self.return_width_or_height(bbox_fruit) == "height" :     # width < length
            # sort according to y coordinates
            fruit_points.sort(key=lambda x: x[1])
        else :                                  # length < width  : 옆으로 누운 것
            # sort according to x coordinates
            fruit_points.sort()
        
        mid_points= self.get_cucumber_mid_points(fruit_points, 10, True)
        
        mid_points_ = mid_points[int(len(mid_points) *(1/4)):]
        first_point_x, first_point_y = mid_points_[0]
        cap_center_x, cap_center_y = self.compute_center_point(self.boxes[inner_idx])
        slope, alpha = get_slope_alpha(first_point_x, first_point_y, cap_center_x, cap_center_y)

        cap_points = self.polygons[inner_idx].copy()
        img = self.img          ### 

        check_boolean = False            
        cap_edge_list = []
        for i in range(self.margin_error):
            for x, y in cap_points:   
                if abs(slope) > 4:
                    if self.return_width_or_height(bbox_fruit) == "height" :
                        if cap_center_x == x: cap_edge_list.append([x, y])
                    else:
                        if cap_center_y == y: cap_edge_list.append([x, y])
                else:
                    cap_edge_list, _, _ = self.find_coordinate_slicing(True, x, y, slope, alpha, cap_edge_list, 1, 1, False, i)
                
                if len(cap_edge_list) == 2  : 
                    
                    if get_length(cap_edge_list[0], cap_edge_list[1]) <5: 
                        cap_edge_list.pop(-1)
                        continue
                    check_boolean = True
                    break
           
            if not check_boolean: continue
            else : break 
                    
        if len(cap_edge_list) < 2: height_points = mid_points
        else: 
            min_length = 100000
            first_point = None
            for point in cap_edge_list:
                length = get_length(self.compute_center_point(bbox_fruit), point)
                if length < min_length:
                    min_length = length
                    first_point = point
            
            mid_points_.insert(0, first_point)
            height_points = mid_points_
        
        max_length = -1
        width_points = None
        for i in range(len(height_points)):
            if i >= int(len(height_points)*(4/5)) or\
                i <= int(len(height_points)*(1/5)): continue        # 양 끝 1/5개수 만큼의 point는 탐색하지 않는다
            point_1_x, point_1_y = height_points[i]
            point_2_x, point_2_y = height_points[i+1] 
            center_point_x, center_point_y = self.compute_center_point([point_1_x, point_1_y, point_2_x, point_2_y])
            slope, alpha = get_slope_alpha(point_1_x, point_1_y, point_2_x, point_2_y)
            inverse_slope = (-1)/slope
            inverse_alpha = center_point_y - center_point_x*inverse_slope
            
            tmp_width_points = None
            check_boolean = False
            for i in range(self.margin_error):
                continue_check, continue_num, width_edge_list, skip_num = create_check_flag(fruit_points_for_search)
                for x, y in fruit_points_for_search:
                    width_edge_list, continue_num, continue_check = self.find_coordinate_slicing(continue_check, x, y, inverse_slope, inverse_alpha, width_edge_list, continue_num, skip_num, False, i)
                    
                    if len(width_edge_list) == 2:
                        tmp_width_points = [width_edge_list[0], width_edge_list[1]]
                        check_boolean = True
                        break
                if not check_boolean: continue
                else : break
                    
            if tmp_width_points is None: continue
            
            width_length = get_length(tmp_width_points[0], tmp_width_points[1])
            if width_length > max_length:
                max_length = width_length
                width_points = tmp_width_points        
        
        if width_points is None: return None, None
        else: 
            width_1, width_2 = width_points
            center_points = self.compute_center_point([width_1[0], width_1[1], width_2[0], width_2[1]])
            return height_points, width_points, center_points         
        
    
    def cucumber_fruit_only(self, bbox_fruit, fruit_points, fruit_points_for_search, fruit_only_info):
        if self.return_width_or_height(bbox_fruit) == "height" :     # width < length
            # sort according to y coordinates
            fruit_points.sort(key=lambda x: x[1])
            fruit_points_for_search.sort()
        else :                                  # length < width  : 옆으로 누운 것
            # sort according to x coordinates
            fruit_points.sort()
            fruit_points_for_search.sort(key=lambda x: x[1])
            
        mid_points_cucumber = self.get_cucumber_mid_points(fruit_points, 10, True)

        fruit_only_info["height"] = mid_points_cucumber
        mid_point_idx = int(len(mid_points_cucumber)/2)
        # mid_points_cucumber 중 mid point 를 기준으로 -1번째 point와 +1번째 point사이의 기울기 계산
        for i, point in enumerate(mid_points_cucumber):
            if i == mid_point_idx:
                center_before_point, center_after_point = mid_points_cucumber[i-1], mid_points_cucumber[i+1] 
                break
        x_1, y_1 = center_before_point
        x_2, y_2 = center_after_point
        slope, _ = get_slope_alpha(x_1, y_1, x_2, y_2)

        if slope == 0:
            inverse_slope = 20
        else: inverse_slope = (-1)/slope
        
        # compute width
        x_min, _, x_max, _ = bbox_fruit
        center_point = self.compute_center_point(bbox_fruit)
        x_mid, y_mid = center_point[0], center_point[1]
        alpha = y_mid - x_mid*(inverse_slope)
        
        width_coordinates = None
        check_boolean = False
        for i in range(self.margin_error):
            continue_check, continue_num, edge_list, skip_num = create_check_flag(fruit_points_for_search)
            for x, y in fruit_points_for_search:
                edge_list, continue_num, continue_check = self.find_coordinate_slicing(continue_check, x, y, inverse_slope, alpha, edge_list, continue_num, skip_num, False, i)

                if len(edge_list) == 2:
                    width_coordinates = [edge_list[0], edge_list[1]]
                    check_boolean = True
                    break
            if not check_boolean: continue
            else : break

        if width_coordinates is None: return None
        
        fruit_only_info["width"] = width_coordinates
        
        # 가장 높은 point와 가장 낮은 point의 수직 거리 계산
        first_point, last_point = fruit_only_info['height'][0], fruit_only_info['height'][-1]
        
        fruit_only_info["depth_point"] = dict()
        fruit_only_info["depth_point"]["first_point"] = first_point
        fruit_only_info["depth_point"]["last_point"] = last_point
        
        fruit_only_info["vertical_point"] = dict()
        fruit_width = int(x_max - x_min)
        if first_point[0] < last_point[0]:  # 오른쪽으로 휘었을 때
            if (self.img.shape[1] // 2) < first_point[0] :   # 화면 우즉에 있을 때
                fruit_only_info["vertical_point"]["first_point"] = (first_point[0] - fruit_width, first_point[1])
                fruit_only_info["vertical_point"]["last_point"] = (first_point[0] - fruit_width, last_point[1])
            else :                                      # 화면 좌측에 있을 때
                fruit_only_info["vertical_point"]["first_point"] = (last_point[0] + fruit_width, first_point[1])
                fruit_only_info["vertical_point"]["last_point"] = (last_point[0] + fruit_width, last_point[1]) 
        else :                              # 왼쪽으로 휘었을 때
            if (self.img.shape[1] // 2) > last_point[0] : # 화면 좌측에 있을 때
                fruit_only_info["vertical_point"]["first_point"] = (first_point[0] + fruit_width, first_point[1])
                fruit_only_info["vertical_point"]["last_point"] = (first_point[0] + fruit_width, last_point[1])
            else :                                  # 화면 우즉에 있을 때
                fruit_only_info["vertical_point"]["first_point"] = (first_point[0] - fruit_width, first_point[1])
                fruit_only_info["vertical_point"]["last_point"] = (first_point[0] - fruit_width, last_point[1]) 

        if "width" not in fruit_only_info.keys() or\
            "height" not in fruit_only_info.keys(): return None

        # 오이의 center는 width point의 중점 
        width_1, width_2 = fruit_only_info["width"][0], fruit_only_info["width"][1]
        fruit_only_info['center'] = [(width_1[0] + width_2[0])//2, (width_1[1] + width_2[1])//2]
        
        return fruit_only_info
            
            
    def onion_fruit_only(self, bbox_fruit, fruit_points, fruit_points_for_search, fruit_only_info):
        fruit_only_info["height"] = [[0, 0], [0, 0]]
        center_point = self.compute_center_point(bbox_fruit)

        max_lagnth = -1
        x_1, y_1 = center_point
        x_2, y_2 = 0, 0
        for point in fruit_points:
            length = get_length(point, center_point)
            if max_lagnth < length:
                max_lagnth = length
                x_2, y_2 = point[0], point[1]

        slope, _ = get_slope_alpha(x_1, y_1, x_2, y_2)

        alpha = y_1 - x_1*(slope)

        check_boolean = False
        for i in range(self.margin_error):
            continue_check, continue_num, edge_list, skip_num = create_check_flag(fruit_points_for_search)
            for x, y in fruit_points_for_search:
                edge_list, continue_num, continue_check = self.find_coordinate_slicing(continue_check, x, y, slope, alpha, edge_list, continue_num, skip_num, False, i)

                if len(edge_list) == 2:
                    width_coordinates = [edge_list[0], edge_list[1]]
                    check_boolean = True
                    break
            if not check_boolean: continue
            else : break

        fruit_only_info["width"] = width_coordinates

        if "width" not in fruit_only_info.keys() or\
            "height" not in fruit_only_info.keys(): return None
        
        return fruit_only_info
            

    def get_only_fruit_points(self, idx):
        fruit_points = self.polygons[idx].copy()
        fruit_points_for_search = self.polygons[idx].copy()
        bbox_fruit = self.boxes[idx]
        
        fruit_only_info = dict(segmentation = fruit_points.copy(),
                                type = self.plant_type,
                                bbox = bbox_fruit)

        if self.plant_type in ["paprika", "melon", "onion", "tomato"] :    # cap이 detect되지 않은 fruit라도 표시
            fruit_only_info = self.common_fruit_only(fruit_points, fruit_points_for_search, fruit_only_info)
        elif self.plant_type in ["cucumber"] :      # cucumber에 대한 fruit length, width 계산
            fruit_only_info = self.cucumber_fruit_only(bbox_fruit, fruit_points, fruit_points_for_search, fruit_only_info)

        elif self.plant_type in ["onion"] : # onion의 fruit length, width 계산
            fruit_only_info = self.onion_fruit_only(bbox_fruit, fruit_points, fruit_points_for_search, fruit_only_info)


        return fruit_only_info


    def compute_fruit_height_onion(self, outer_idx, inner_idx):
        cap_points = self.polygons[inner_idx].copy()
        x_center_cap, y_center_cap = self.compute_center_point(self.boxes[inner_idx])
        
        fruit_points =self.polygons[outer_idx].copy()
        fruit_bbox_width, fruit_bbox_height = self.compute_width_height(self.boxes[outer_idx])
        max_length = -1
        for fruit_x, fruit_y in fruit_points:
            # cap의 center cooridnate와 fruit의 각 boundary coordinate사이의 거리를 계산
            length = get_length([fruit_x, fruit_y], [x_center_cap, y_center_cap])

            # 두 point사이의 거리가 fruit Bbox의 width 또는 height의 절반보다 작으면 continue
            if length < fruit_bbox_width/2 or length < fruit_bbox_height/2: continue   
            
                
            if max_length < length : 
                max_length = length
                fruit_x_coord, fruit_y_coord = fruit_x, fruit_y
        
        # cap의 center coordinate를 지나는 slope의 기울기를 가진 1차함수
        # 기울기 : cap과 segmentation을 기준으로  
        slope, alpha = get_slope_alpha(fruit_x_coord, fruit_y_coord, x_center_cap, y_center_cap)
        
        # fruit_x_coord, fruit_y_coord 좌표를 지나고 기울기 slope을 가진 1차함수 위의 점 중 fruit의 boundary point를 찾는다. 
        continue_check, continue_num, edge_list, skip_num = create_check_flag(cap_points)    
        check_boolean = False            
        
        for i in range(self.margin_error):
            for x, y in cap_points:   
                edge_list, continue_num, continue_check = self.find_coordinate_slicing(continue_check, x, y, slope, alpha, edge_list, continue_num, skip_num, False, i)
                
                if len(edge_list) == 2  : 
                    if edge_list[0] == edge_list[1] : 
                        edge_list.pop(-1)
                        continue
                   
                    length_1 = get_length(edge_list[0], [fruit_x_coord, fruit_y_coord])
                    length_2 = get_length(edge_list[1], [fruit_x_coord, fruit_y_coord])
                    check_boolean = True

                    
                    if length_1 <= length_2:
                        return [edge_list[0], [fruit_x_coord, fruit_y_coord]]
                    else: 
                        return [edge_list[1], [fruit_x_coord, fruit_y_coord]]
                    break 
            if not check_boolean: continue
            else : break    
            
                      
        # if abs(slope) > 20:  # boundary coordinates중 center coordinate와 같은 x값을 가진coordinate를 구한다.
        #     for x, y, in cap_coordinates:
        #         if continue_check:
        #             if x == x_center_cap:
        #                 edge_list.append([x, y])
        #                 continue_check = False
        #         else:
        #             if continue_num == 0:
        #                 continue_num = skip_num
        #                 continue_check = True
        #             else:
        #                 continue_num -=1 
                
        #         if len(edge_list) == 2:
        #             length_1 = math.sqrt(math.pow(edge_list[0][0] - fruit_x_coord, 2) + math.pow(edge_list[0][1] - fruit_y_coord, 2))
        #             length_2 = math.sqrt(math.pow(edge_list[1][0] - fruit_x_coord, 2) + math.pow(edge_list[1][1] - fruit_y_coord, 2))
                
        #             if length_1 <= length_2:
        #                 tmp_fruit_dict["height"] = [edge_list[0], [fruit_x_coord, fruit_y_coord]]
        #             else: 
        #                 tmp_fruit_dict["height"] = [edge_list[1], [fruit_x_coord, fruit_y_coord]]
        #             break
        # else :  # boundary coordinate를 탐색하며 y = slope*x + alpha 에 만족하는 y, x coordinate를 찾는다.
        #     check_boolean = False
        #     for i in range(self.margin_error):
        #         for x, y in cap_coordinates:   
        #             edge_list, continue_num, continue_check = self.find_coordinate_slicing(continue_check, x, y, slope, alpha, edge_list, continue_num, skip_num, False, i)
                    
        #             if len(edge_list) == 2: 
        #                 length_1 = math.sqrt(math.pow(edge_list[0][0] - fruit_x_coord, 2) + math.pow(edge_list[0][1] - fruit_y_coord, 2))
        #                 length_2 = math.sqrt(math.pow(edge_list[1][0] - fruit_x_coord, 2) + math.pow(edge_list[1][1] - fruit_y_coord, 2))
        #                 check_boolean = True

        #                 if length_1 <= length_2:
        #                     tmp_fruit_dict["height"] = [edge_list[0], [fruit_x_coord, fruit_y_coord]]
        #                 else: 
        #                     tmp_fruit_dict["height"] = [edge_list[1], [fruit_x_coord, fruit_y_coord]]
        #                 break 
        #         if not check_boolean: continue
        #         else : break
        
        return None 

    def compute_fruit_height_paprika(self, outer_idx, inner_idx, is_cap, is_cap_2):
        # sub_function
        def get_cap_slope(x_center_cap, y_center_cap, cap_points, fruit_points):
            # cap의 center point 로부터 가장 먼 cap point 찾기
            max_length_cap = -1
            for x, y in cap_points: 
                length_cap = get_length([x_center_cap, y_center_cap], [x, y])
                if max_length_cap < length_cap:
                    max_length_cap = length_cap
                    cap_x, cap_y = x, y
            # `cap의 center point`와 `cap의 center point 로부터 가장 먼 cap point`사이의 기울기 
            slope_cap, _ = get_slope_alpha(cap_x, cap_y, x_center_cap, y_center_cap)
            if slope_cap == 0:  # 수평인 경우, fruit가 수직으로 똑바로 열려있다고 가정
                # inverse_slope_cap: 위 기울기의 수직인 직선의 기울기
                inverse_slope_cap = 100
            else :              
                inverse_slope_cap = (-1/slope_cap)      
            inverse_alpha_cap = y_center_cap - inverse_slope_cap * x_center_cap

            # slope_fruit: `cap의 center point`와 `cap의 center point 로부터 가장 먼 fruit point`사이의 기울기 
            max_length_fruit = -1
            for x, y in fruit_points: 
                length_fruit = get_length([x_center_cap, y_center_cap], [x, y])                
                if max_length_fruit < length_fruit:    
                    max_length_fruit = length_fruit
                    fruit_x, fruit_y = x, y
            slope_fruit, _ = get_slope_alpha(fruit_x, fruit_y, x_center_cap, y_center_cap)

            slope = compute_median_slope(inverse_slope_cap, inverse_alpha_cap, slope_fruit, [x_center_cap, y_center_cap])       
            alpha = y_center_cap - x_center_cap*slope
            
            return slope, alpha
        
        # sub_function
        def compute_median_slope(slope_1, slope_1_alpha, slope_2, datum_point):
            x, y = datum_point
            # slope: 두 기울기의 중간값
            if slope_1 * slope_2 > 0:     # 두 기울기가 같은 방향으로 기울어진 경우
                # slope_1와 slope_2의 중간 기울기를 가진 기울기 계산
                slope = (slope_1 + slope_2)/2
            else:                                       # 두 기울기가 서로 다른 방향으로 기울어진 경우(대칭처럼)
                # cap의 y좌표에 10을 더한 후, `해당 point`와 `datum_point`사이의 기울기 계산
                # tmp_y = y + 10
                # alpha_fruit = y - slope_2 * x
                # tmp_x_fruit = (tmp_y - alpha_fruit)/slope_2
                # tmp_x_cal = (tmp_y - slope_1_alpha)/slope_1
                # slope, _ = get_slope_alpha(x, y, (tmp_x_cal + tmp_x_fruit)/2, tmp_y)
                
                # 그냥 절대값 합산
                slope = abs(slope_1) + abs(slope_2)
            return slope
        
        # main
        if is_cap and is_cap_2:
            cap_2_idx = inner_idx[1]
            cap_1_idx = inner_idx[0]
            inner_idx = cap_1_idx
        

        bbox_cap = self.boxes[inner_idx].copy()
        cap_points = self.polygons[inner_idx].copy()
        x_center_cap, y_center_cap = self.compute_center_point(bbox_cap)
        bbox_fruit = self.boxes[outer_idx].copy()
        fruit_points = self.polygons[outer_idx].copy()
        
       
            
        x_center_fruit, y_center_fruit = self.compute_center_point(bbox_fruit)
        fruit_bbox_width, fruit_bbox_height = self.compute_width_height(bbox_fruit)
     
        if is_cap and is_cap_2:             # cap_1과 cap_2 모두를 포함한 fruit인 경우
            x_center_cap_2, y_center_cap_2 = self.compute_center_point(self.boxes[cap_2_idx])
            slope_1, alpha_1  = get_slope_alpha(x_center_fruit, y_center_fruit, x_center_cap_2, y_center_cap_2)
            slope_2, _ = get_cap_slope(x_center_cap, y_center_cap, cap_points, fruit_points)                    
            slope = compute_median_slope(slope_1, alpha_1, slope_2, [x_center_cap, y_center_cap])  
        
            alpha = y_center_cap - x_center_cap*slope                                               
        elif not is_cap and is_cap_2:       # cap_2만을 포함한 fruit인 경우
            slope, alpha  = get_slope_alpha(x_center_fruit, y_center_fruit, x_center_cap, y_center_cap)

        else:                               # cap만을 포함한 fruit인 경우
            slope, alpha = get_cap_slope(x_center_cap, y_center_cap, cap_points, fruit_points)
        
       
        if abs(slope) > 15:         # 기울기가 너무 높은 경우 fruit의 height는 수직의 직선으로 계산한다
            fruit_point_bottom_y, cap_point_bottom_y = -1, -1
            for fruit_x, fruit_y in fruit_points:
                if x_center_cap == fruit_x:
                    if fruit_point_bottom_y < fruit_y : 
                        fruit_point_bottom_y = fruit_y
            for cap_x, cap_y in cap_points: 
                if x_center_cap == cap_x:
                    if cap_point_bottom_y < cap_y : 
                        cap_point_bottom_y = cap_y
            
            return [[x_center_cap, cap_point_bottom_y], [x_center_cap, fruit_point_bottom_y]]
            
        elif abs(slope) < 0.067:    # 기울기가 너무 낮은 경우 fruit의 height는 수평의 직선으로 계산한다
            # TODO: cap이 fruit의 좌측에 위치하는지, 우측에 위치하는지 확인 후 case에 따라 code작성
            return None
        else:                           # 기울기가 너무 높지 않은 경우
            # cap : fruit의 bottom point, top point를 각각 따로 찾는다.
            # cap_2 : fruit의 bottom point, top point를 한 번에 찾는다.
            
            # fruit의 bottom point찾기
            edge_list = []                     
            check_boolean_fruit = False                  
            is_cap_2_fruit_dict = None
            for i in range(self.margin_error):  
                for fruit_x, fruit_y in fruit_points:                            
                    edge_list, _, _ = self.find_coordinate_slicing(True, fruit_x, fruit_y, slope, alpha, edge_list, 0, 0, False, i)
                    
                    if len(edge_list) == 2  : 
                        if edge_list[0] == edge_list[1]:    # 두 점이 같은 점이면 continue
                            edge_list.pop(-1)
                            continue
                        
                        # 두 점 사이가 fruit_bbox_height/3 보다 낮으면 잘못된 계산, continuie
                        length = get_length(edge_list[0], edge_list[1])
                        if length < fruit_bbox_height/3 :
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
                        
                        
                        if length_1 <= length_2: edge_list.pop(0) 
                        else: edge_list.pop(-1)
                            
                        break 
                if check_boolean_fruit: break
                else : continue  
                
            if len(edge_list) == 0 :return None     # height좌표를 구하지 못한 경우
            
     
            if is_cap_2 and is_cap_2_fruit_dict is None : 
                # point를 한 개 밖에 찾지 못한 경우
                for fruit_x, fruit_y in fruit_points:   
                    # fruit의 하단 point중 x좌표가 비슷한 point를 반대편 point로 결정
                    if 3> abs(edge_list[0][0] - fruit_x) \
                        and fruit_bbox_height/3 < get_length(edge_list[0], [fruit_x, fruit_y]):
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

            elif (not is_cap_2 and is_cap) or\
                (is_cap_2 and is_cap) : 
                    # cap_2는 없고 cap만 있는 경우 or # cap, cap_2 전부 있는 경우
                if  fruit_bbox_height/3 > get_length(edge_list[0], [x_center_cap, y_center_cap]):
                    # fruit의 bottom point와 cap의 center point간의 거리가 
                    # fruit_bbox_height/3 보다 작다: fruit의 상단 부근의 point라는 뜻. 
                    for fruit_x, fruit_y in fruit_points:    
                        # fruit의 하단 point중 x좌표가 동일한 point를 bottom point로 결정
                        if edge_list[0][0] == fruit_x and (fruit_y - edge_list[0][1]) > fruit_bbox_height/3:
                            edge_list = [[int(fruit_x), int(fruit_y)]]

                fruit_point_bottom = edge_list[0]
                
                # fruit의 top point를 찾는다.
                # cap인 경우 cap point중에서 top point를 찾아야 한다.
                edge_list_cap = [] 
                check_boolean_cap = False
                slope_fruit, alpha_fruit = get_slope_alpha(edge_list[0][0], edge_list[0][1], x_center_cap, y_center_cap) 
                for i in range(self.margin_error):     
                    for cap_x, cap_y in cap_points: 
                        edge_list_cap, _, _ = self.find_coordinate_slicing(True, cap_x, cap_y, slope_fruit, alpha_fruit, edge_list_cap, 0, 0, False, i)
                        if len(edge_list_cap) == 2  : 
                            if edge_list_cap[0] == edge_list_cap[1]:    # 두 point가 같은 값일 경우
                                edge_list_cap.pop(0)
                                continue
                            
                            length = get_length(edge_list_cap[0], edge_list_cap[1])
                            if length < abs(fruit_bbox_height)/4 :  # fruit bbox의 1/4보다 작은 경우: 두 point가 가까이 있을 경우
                                if get_length(edge_list_cap[0], [x_center_fruit, y_center_fruit]) >\
                                    get_length([x_center_fruit, y_center_fruit], [x_center_cap, y_center_cap]):
                                    # 두 점 모두 fruit의 center로부터 cap_center보다 멀리 떨어져있으면 찾는 point가 아니다.
                                    edge_list_cap = []
                                    continue
                                        
                                
                            length_1 = get_length(edge_list_cap[0], fruit_point_bottom)    # 첫 번째 point와 fruit_bottom point사이의 거리
                            length_2 = get_length(edge_list_cap[1], fruit_point_bottom)    # 두 번째 point와 fruit_bottom point사이의 거리
                            check_boolean_cap = True    
                            
                            if length_1 <= length_2: 
                                edge_list_cap.pop(-1) # length_2가 크다: edge_list_cap[0]가 더 가까이있는 point == top point
                            else: 
                                edge_list_cap.pop(0)
                            break
                    if check_boolean_cap: break
                    else : continue   
                
                if len(edge_list_cap) == 0: return None
                fruit_point_top = edge_list_cap[0]

                return [fruit_point_top, fruit_point_bottom]

                
                
                
                        
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
            if self.plant_type == 'melon' : 
                # 가장 낮은 y좌표들 중 x 기준 중점을 bottom point로 사용
                bottom_point_list = []
                max_y_coordinate = -1
                for point in fruit_points_part_2:
                    if max_y_coordinate <= point[1] : 
                        max_y_coordinate = point[1]
                        bottom_point_list.append(point)
                
                if len(bottom_point_list) == 1:
                    bottom_point = bottom_point_list[0]
                elif len(bottom_point_list) >= 2:
                    bottom_point_list.sort()
                    bottom_point = bottom_point_list[len(bottom_point_list)//2]
            elif self.plant_type == 'tomato':
                # cap의 center point로부터 가장 먼 fruit point가 bottom point
                fruit_points_part_2 = fruit_points[int(len(fruit_points)*2/3):]   
                max_length = -1
                for point in fruit_points_part_2:  # cap의 center point로부터 가장 가까운 fruit point가 top point
                    tmp_length = get_length([x_center_cap, y_center_cap], point)
                    if max_length < tmp_length:
                        bottom_point = point
                        max_length = tmp_length  
                        
            return bottom_point
        
        
        # main
        fruit_info = dict()
        
        x_center_cap, y_center_cap = self.compute_center_point(self.boxes[inner_idx])
        
        fruit_bbox_width, fruit_bbox_height = self.compute_width_height(self.boxes[outer_idx])
        fruit_points = self.polygons[outer_idx].copy()
        fruit_points.sort(key = lambda x:x[1])

        fruit_points_part_1 = fruit_points[:int(len(fruit_points)/3)]
        fruit_points_part_2 = fruit_points[int(len(fruit_points)*2/3):]                    

        top_point = get_top_point(fruit_points_part_1, x_center_cap, y_center_cap)
        bottom_point =  get_bottom_point(fruit_points, fruit_points_part_2, x_center_cap, y_center_cap)
           

        if top_point is None or bottom_point is None: return None
        
        fruit_info["height"] = [top_point, bottom_point]
        width_center_point = [int((top_point[0] + bottom_point[0])/2), int((top_point[1] + bottom_point[1])/2)]
        fruit_info["center"] = width_center_point

        slope, _ = get_slope_alpha(top_point[0], top_point[1], bottom_point[0], bottom_point[1]) 
        
        # ----
        width_point = []
        inverse_slope = -1/(slope)
        inverse_alpha = -1*width_center_point[0]*inverse_slope + width_center_point[1]
        check_boolean = False
        for i in range(3):  
            continue_check, _, edge_list, skip_num = create_check_flag(fruit_points)
            continue_num = self.margin_error + 10
            for x, y in fruit_points:
                edge_list, continue_num, continue_check = self.find_coordinate_slicing(continue_check, x, y, inverse_slope, inverse_alpha, edge_list, continue_num, skip_num, False, i)
                
                if len(edge_list) == 2:
                    fruit_length = math.sqrt(math.pow(edge_list[0][0] - edge_list[1][0], 2) + math.pow(edge_list[0][1] - edge_list[1][1], 2))
                    if fruit_length < fruit_bbox_width/2 :    # 두 점 사이의 거리가 짧으면 continue(최소 fruit bbox의 절반은 되어야 한다.)
                        edge_list.pop(-1)
                        continue
                    width_point = [edge_list[0], edge_list[1]]
                    check_boolean = True
                    break
            if not check_boolean: continue
            else : break
        
        if len(width_point) == 0: 
            width_point = []
            fruit_coordinates_part_3 = fruit_points[int(len(fruit_points)*1/3):int(len(fruit_points)*2/3)]
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
            
        if "width" not in fruit_info.keys() or\
            "height" not in fruit_info.keys(): return None
        return fruit_info
                
    def compute_fruit_height_strawberry(self, outer_idx, inner_idx):
        fruit_points = self.polygons[outer_idx].copy()
        x_center_cap, y_center_cap = self.compute_center_point(self.boxes[inner_idx])
        
        tmp_fruit_points =  fruit_points.copy()
        tmp_fruit_points.sort(key=lambda x : x[1])
        fruit_points_top = tmp_fruit_points[:int(len(tmp_fruit_points)/2)]       # 가까운 point를 찾을땐 fruit의 상단 boundary points만 사용
        fruit_points_bottom = tmp_fruit_points[int(len(tmp_fruit_points)/2):]    # 먼 point를 찾을땐 fruit의 하단 boundary points만 사용

        min_length = self.img.shape[1]
        for fruit_x, fruit_y in fruit_points_top:  # center coordinate of cap 으로부터 가장 가까운 boundary coordinates of fruit의 한 point를 찾아낸다.
            # cap의 center cooridnate와 fruit의 각 boundary coordinate사이의 거리를 계산
            length = math.sqrt(math.pow(fruit_x - x_center_cap, 2) + math.pow(fruit_y - y_center_cap, 2))

            if length < min_length:
                min_length = length
                fruit_x_coord, fruit_y_coord = fruit_x, fruit_y

        max_length = -1
        for fruit_x, fruit_y in fruit_points_bottom:
            length = math.sqrt(math.pow(fruit_x - fruit_x_coord, 2) + math.pow(fruit_y - fruit_y_coord, 2))

            if length > max_length:
                max_length = length
                height_x, height_y = fruit_x, fruit_y

        return [[fruit_x_coord, fruit_y_coord], [height_x, height_y]] 
    
    
    def compute_fruit_width(self, outer_idx, fruit_info): 
        bbox_fruit = self.boxes[outer_idx]
        fruit_points = self.polygons[outer_idx].copy()
        
        # height를 표현하는 point 2개의 중점에서의 1차함수 기울기, 절편
        
        coordinate_1, coordinate_2 = fruit_info["height"]
        [x_1, y_1] = coordinate_1     
        [x_2, y_2]  = coordinate_2
        
        # x_mid, y_mid의 위치에서 inverse_slope의 기울기를 가지며 alpha의 절편을 가진 1차함수 계산 
        slope, _ = get_slope_alpha(x_1, y_1, x_2, y_2) 
        
        if slope == 0:
            inverse_slope = 20
        elif abs(slope) > 20 :   # 기울기의 절대값이 20보다 크면 폭을 수평으로 계산한다.
            inverse_slope = 0
        else: inverse_slope = (-1)/slope
        
        mid_point_alpha = []
        if self.plant_type in ["paprika", "strawberry"]:
            mid_point_num = 5
            x_val, y_val = abs(x_1-x_2)/mid_point_num, abs(y_1-y_2)/mid_point_num 
            num_list = [i - mid_point_num//2 for i in range(mid_point_num)]     
            for i in num_list:
                x_mid, y_mid = (x_1+x_2)/2 + (x_val*i), (y_1+y_2)/2 + (y_val*i)
                alpha = y_mid - x_mid*(inverse_slope)
                mid_point_alpha.append([alpha, [int(x_mid), int(y_mid)]])
        elif self.plant_type == "onion":        
            x_min_fruit, y_min_fruit, x_max_fruit, y_max_fruit = bbox_fruit
            center_point = (int((x_min_fruit + x_max_fruit)/2),int((y_min_fruit + y_max_fruit)/2)) 
            # fruit_points중 fruit의 center에서 가장 먼 point와의 기울기를 계산
            max_lagnth = -1
            x_1, y_1 = center_point
            x_2, y_2 = 0, 0
            for point in fruit_points:
                length = math.sqrt(math.pow(point[0] - center_point[0], 2) + math.pow(point[1] - center_point[1], 2))
                if max_lagnth < length:
                    max_lagnth = length
                    x_2, y_2 = point[0], point[1]
            slope, _ = get_slope_alpha(x_1, y_1, x_2, y_2)     
            inverse_slope = slope       # 편의를 위해 name만 바꿈
            alpha = y_1 - x_1*(slope)   # fruit의 center point에서의 alpha
            mid_point_alpha.append([alpha, list(center_point)])
            
            
        width_length_list = []
        for mid_point in mid_point_alpha:
            alpha = mid_point[0]

            width_coordinates = None
            # fruit boundary coordinates위의 한 coordinate를 구한다.
            check_boolean = False
            for i in range(self.margin_error):  
                continue_check, continue_num, edge_list, skip_num = create_check_flag(fruit_points)
                for x, y in fruit_points:
                    edge_list, continue_num, continue_check = self.find_coordinate_slicing(continue_check, x, y, inverse_slope, alpha, edge_list, continue_num, skip_num, False, i)

                    if len(edge_list) == 2:
                        width_coordinates = [edge_list[0], edge_list[1]]
                        check_boolean = True
                        break
                if not check_boolean: continue
                else : break

            
            if width_coordinates is None: continue
            length = get_length(width_coordinates[0], width_coordinates[1])
            
            width_length_list.append([length, width_coordinates, mid_point[1]])
        
        if len(width_length_list) == 0: return fruit_info
        
        
        max_length = -1
        for width_info in width_length_list:
            width_length, width_coordi, mid_center_point = width_info
                                
            if max_length<width_length:
                max_length = width_length
                fruit_info["width"] = width_coordi
                fruit_info["center"] = mid_center_point
                
        return fruit_info                     

    def get_cap_fruit_points_(self, outer_idx, inner_idx, fruit_point_dict,
                             is_cap, is_cap_2):
        # sub_function  (과의 꼭지가 있을 경우에, 꼭지가 fruit영역 안의 위치에 따라(중앙 or 가장자리) 적용 algorithm이 달라진다.)
        def check_cap_location(inner_idx_):
            # cap이 fruit 영역 안의 중앙 부근에 위치했는지 확인.  (대부분 cap이 fruit의 가장자리 위치에 있음)
            center_flag = False
            bbox_fruit = self.boxes[outer_idx].copy()
            center_fruit = self.compute_center_point(bbox_fruit)
            width_fruit, height_fruit = self.compute_width_height(bbox_fruit)

            center_aredict_fruit = dict(x_min = int(center_fruit[0] - width_fruit/5),
                                        y_min = int(center_fruit[1] - height_fruit/5),
                                        x_max = int(center_fruit[0] + width_fruit/5),
                                        y_max = int(center_fruit[1] + height_fruit/5))
    
            center_cap = self.compute_center_point(self.boxes[inner_idx_])
            center_cap_x, center_cap_y = center_cap
            
            if center_aredict_fruit['x_min'] < center_cap_x and\
                center_aredict_fruit['y_min'] < center_cap_y and\
                center_aredict_fruit['x_max'] > center_cap_x and\
                center_aredict_fruit['y_max'] > center_cap_y:
                center_flag = True
            return center_flag
        
        
        
        # main
        if is_cap and is_cap_2: 
            center_flag = check_cap_location(inner_idx[0])
        else:
            center_flag = check_cap_location(inner_idx)
        fruit_info = dict(segmentation = self.polygons[outer_idx].copy(),
                          type = self.plant_type,
                          bbox = self.boxes[outer_idx])
        
        if self.plant_type == "onion":
            # onion은 cap이 fruit의 center box에 포함되든 아니든 같은 방식으로 계산
            fruit_info['height'] = self.compute_fruit_height_onion(slope, outer_idx, inner_idx)
            fruit_point_dict['cap_fruit_side'].append(fruit_info)
            return fruit_point_dict
                
        if not center_flag: # cap이 fruit의 center box에 포함되지 않은 경우(가장자리에 위치한 경우)
            # height계산
            if self.plant_type == "paprika":
                fruit_height = self.compute_fruit_height_paprika(outer_idx, inner_idx, is_cap, is_cap_2)      
                   
                if fruit_height is not None:
                    fruit_info['height'] = fruit_height
            elif self.plant_type in ["melon", "tomato"]:
                # width, height, center전부 계산
                fruit_info_dict = self.compute_fruit_w_h_common_fruit(outer_idx, inner_idx)
                if fruit_info_dict is not None:
                    for key, item in fruit_info_dict.items():
                        fruit_info[key] = item
            elif self.plant_type == "strawberry" :
                fruit_info['height'] = self.compute_fruit_height_strawberry(outer_idx, inner_idx)
            elif self.plant_type == "cucumber":     # width, height, center전부 계산
                fruit_height, fruit_width, fruit_center = self.cucumber_fruit_cap(outer_idx, inner_idx)    
                if fruit_height is not None:
                    fruit_info['height'], fruit_info['width'], fruit_info['center'] = fruit_height, fruit_width, fruit_center
            
                    
            if "height" not in list(fruit_info.keys()) : return fruit_point_dict
            
            # width 계산
            if self.plant_type in ['paprika', 'strawberry']:
                fruit_info = self.compute_fruit_width(outer_idx, fruit_info)
                
            if "width" not in list(fruit_info.keys()) : return fruit_point_dict
            
            fruit_point_dict['cap_fruit_side'].append(fruit_info)
            return fruit_point_dict
            

        # cap이 fruit의 center box에 포함되는 경우
        # width만 구할 수 있다.           
        elif center_flag:
            
            fruit_info['height'] = list() 
            if self.plant_type in ["paprika", "melon"]:     # height계산 안함.
                x_center_cap, y_center_cap = self.compute_center_point(self.boxes[inner_idx])
                fruit_points = self.polygons[outer_idx].copy()
                lowst_lenth = 100000

                # for calculate fruit width
                # length_difference_list : [[length_difference_1, [x_a_1, y_a_1, x_b_1, y_b_1]], [length_difference_2, [x_a_2, y_a_2, x_b_2, y_b_2]], ...] 
                length_difference_list = []

                for fruit_x, fruit_y in fruit_points:
                    ## for calculate fruit width
                    slope, alpha = get_slope_alpha(fruit_x, fruit_y, x_center_cap, y_center_cap)
                    # cap의 center coordinate를 지나는 1차함수 위에 존재하는 fruit coordinates를 찾으면 두 점 사이의 거리를 구한 후 
                    # 기존에 구한 length_1와의 차이값을 length_difference_list에 할당
                    # 오차범위 2 
                    length_1 = get_length([fruit_x, fruit_y], [x_center_cap, y_center_cap])
                    for x, y in fruit_points:
                        if int(y) >= int(slope*x + alpha) - 1 and int(y) <= int(slope*x + alpha) + 1 and (fruit_x != x and fruit_y !=y):
                            length_2 = get_length([x, y], [x_center_cap, y_center_cap])
                            length_difference_list.append([abs(length_1-length_2), [fruit_x, fruit_y, x, y]])

                for length_difference, coordinates_two_point in length_difference_list:
                    if lowst_lenth > length_difference:
                        lowst_lenth = length_difference
                        width_fruit_coordinates = [[coordinates_two_point[0], coordinates_two_point[1]], [coordinates_two_point[2], coordinates_two_point[3]]] 
                
                # width의 양 끝점 좌표
                width_points_list = get_width_point(width_fruit_coordinates, 5)

                if len(width_points_list) !=0:
                    fruit_info["width"] = width_points_list  
                    fruit_info["center"] = list(width_points_list[len(width_points_list)//2])
            
                if "width" not in fruit_info.keys() : return fruit_point_dict
                fruit_point_dict["cap_fruit_above"].append(fruit_info)

        
        return fruit_point_dict                     



##### leaf와 midrid에 관련된 methods

    def get_center_coordinates(self, leaf_coordinates, width_or_height):
        if width_or_height == "width":
            leaf_coordinates.sort()
        else:
            # 각 leaf_coordinates_half의 bbox를 계산 후 한번 더 sort
            leaf_coordinates.sort(key= lambda x: x[1])
            
        midrid_center_coordinates = []
    
        x_coordinates, y_coordinates = [], []
        for point in leaf_coordinates : 
            x_coordinates.append(point[0])
            y_coordinates.append(point[1])      

        temp_list_2 = self.x_or_y_center_coordinates(x_coordinates, y_coordinates, width_or_height) 

        list_for_check_redundant = []
        total_other_value = None
        count = None

        for idx_, center_coor_no_dupl in enumerate(temp_list_2):
            if width_or_height == "height":
                if center_coor_no_dupl[1] not in list_for_check_redundant:      
                    # 이미 check한 y좌표가 아니라면
                    list_for_check_redundant.append(center_coor_no_dupl[1])

                    if idx_ == 0 : 
                        total_other_value = center_coor_no_dupl[0]
                        count = 1
                        continue

                    if count == 1:      #   y coordiante 에 대응되는 x coodinate가 1개인 경우
                        midrid_center_coordinates.append(center_coor_no_dupl)

                    else:               #   y coordiante 에 대응되는 x coodinate가 2개 이상인 경우
                        midrid_center_coordinates.append([int(total_other_value / count), center_coor_no_dupl[1]])
                        # int(total_x_value / count) == mean of x coordinate
                        # center_coor_no_dupl[1]  == y coordinate

                    total_other_value = center_coor_no_dupl[0]
                    count = 1
                else :      
                    # 이미 check한 y좌표라면, x좌표 값을 add
                    total_other_value += center_coor_no_dupl[0]
                    count +=1
            else:
                if center_coor_no_dupl[0] not in list_for_check_redundant:      
                    # 이미 check한 y좌표가 아니라면
                    list_for_check_redundant.append(center_coor_no_dupl[0])

                    if idx_ == 0 : 
                        total_other_value = center_coor_no_dupl[1]
                        count = 1
                        continue

                    if count == 1:      #   y coordiante 에 대응되는 x coodinate가 1개인 경우
                        midrid_center_coordinates.append(center_coor_no_dupl)

                    else:               #   y coordiante 에 대응되는 x coodinate가 2개 이상인 경우
                        midrid_center_coordinates.append([center_coor_no_dupl[0], int(total_other_value / count)])
                        # int(total_x_value / count) == mean of x coordinate
                        # center_coor_no_dupl[1]  == y coordinate

                    total_other_value = center_coor_no_dupl[1]
                    count = 1
                else :      
                    # 이미 check한 y좌표라면, x좌표 값을 add
                    total_other_value += center_coor_no_dupl[1]
                    count +=1

        return midrid_center_coordinates



    def select_valiable_point(self, center_coordinates, threshold_length, using_point_ratio):
        # 거리가 threshold_length이상인 point는 전부 지우고, 
        # 전체 point중 (using_point_ratio-1)/using_point_ratio비율 만큼 버린다.
        
        point_list = []
        for i, point_i in enumerate(center_coordinates) :
            if i == len(center_coordinates)-1 or i == 0: 
                point_list.append(point_i)

            if using_point_ratio != 1:                          # using_point_ratio = 1 이면 버리는 point는 없다.
                if (i+1) % using_point_ratio !=0 : continue       # using_point_ratio = 3 이면 2/3 개수만큼은 버린다. 
                     

            minimum_point = None
            min_length = 1000000
            
            for point_j in center_coordinates[i+1:] :
                length = math.sqrt(math.pow(point_i[0] - point_j[0], 2) + math.pow(point_i[1] - point_j[1], 2))
                if min_length > length:
                    min_length = length
                    minimum_point = point_j
            
            if minimum_point is not None:
                if min_length < threshold_length:   # threshold_length = 5 이면 5보다 크거나 같은 거리의 point는 버린다.
                    point_list.append(minimum_point)
        
        return point_list



    def get_onion_leaf_info(self):
        point_dict_list = []
        
        for idx in self.get_object_idx("leaf"):
            onion_leaf_info = {}
            leaf_points = self.polygons[idx].copy()
            bbox_leaf = self.boxes[idx]
            onion_leaf_info["segmentation"] = leaf_points.copy()
            onion_leaf_info["type"] = self.plant_type
            onion_leaf_info["bbox"] = self.boxes[idx]
            
            # onion의 leaf길이(length)
            sorted_leaf_center_points = self.get_center_coordinates(leaf_points, 
                                                                    self.return_width_or_height(bbox_leaf))
            
            num_spot = self.config.NUM_SPOT       # c_cfg.NUM_SPOT == 10 이면 10개의 point를 찍는다.

            count_bot = 1
            point_list = []
            ### select particular point
            for i, center_coordinate in enumerate(sorted_leaf_center_points):
                if i  == int(len(sorted_leaf_center_points)*(count_bot/num_spot)) or i == 0 or i == (len(sorted_leaf_center_points) -1):    
                    count_bot +=1
                    point_list.append(center_coordinate)

            # point_list의 첫 번째 point와 마지막 point사이의 기울기 값을 통해 onion의 leaf가 수직으로 자랏는지, 수평으로 자랐는지 유로
            if len(point_list) == 0: continue
            x_first, y_first = point_list[0][0], point_list[0][1]
            x_last, y_last = point_list[-1][0], point_list[-1][1]
            midrid_slope, _ = get_slope_alpha(x_first, y_first, x_last, y_last)        # midrid의 첫 번째 point와 마지막 point사이의 기울기
            if abs(midrid_slope) > 7 : point_list.sort(key= lambda x: x[1])
            elif abs(midrid_slope) < 0.15 : point_list.sort()
            leaf_center_points = self.parsing_onion_leaf_points(point_list)  
            
            # onion의 leaf굵기(width) 계산
            onion_leaf_info["leaf_width_edge_coordinates"] = []
            edge_point_list = []
            for idx in range(len(leaf_center_points)):
                if idx == 0 or idx == len(leaf_center_points)-1 : continue

                x_this, y_this = leaf_center_points[idx][0], leaf_center_points[idx][1]
                x_next, y_next = leaf_center_points[idx+1][0], leaf_center_points[idx+1][1]
                                    
                slope, _ = get_slope_alpha(x_this, y_this, x_next, y_next) 
                if slope == 0: continue         # 기울기가 0인경우는 잘못 계산된 것
                        
                if  (abs(slope) > 20): slope = 100             # point의 기울기가 너무 높으면 잎이 수직으로 찍힌 경우. width를 수평으로 긋는다. 
                elif (abs(slope) < 1/20) : slope = 1/100        # point의 기울기가 너무 낮으면 잎이 수평으로 찍힌 경우. width를 수직으로 긋는다.
                elif (midrid_slope * slope < 0) : continue       # 위 두 조건을 만족하지 않은 경우, 두 기울기는 서로 같은 음수 or 양수 관계여야 한다. 
        
                x_mid, y_mid = (x_this+x_next)//2, (y_this+y_next)//2       # midrid위의 각 point 사이의 중점     
                coordinate_two_point = [[x_this, y_this], [x_mid, y_mid]]   # midrid위의 현재 point와, 중점 point
                inverse_slope = (-1)/slope

                for point in coordinate_two_point:                                                  
                    x_coordinate, y_coordinate = point
                    alpha = y_coordinate - x_coordinate*(inverse_slope)   # 위에서 계산한 중점에서의 1차 함수의 절편   
        
                    check_boolean = False 
                    for i in range(self.margin_error):
                        spot_list = []
                        for x, y in leaf_points:
                            
                            if abs(inverse_slope) < 1/20:  
                                if y_coordinate == y : spot_list.append([x, y])
                            elif abs(inverse_slope) > 20: 
                                if x_coordinate == x : spot_list.append([x, y])
                                
                            else: spot_list, _, _ = self.find_coordinate_slicing(True, x, y, inverse_slope, alpha, spot_list, 0, 3, 
                                                                                                        False, i)
                                
                            if len(spot_list) == 2:
                                length = math.sqrt(math.pow(spot_list[0][0] - spot_list[1][0], 2) + math.pow(spot_list[0][1] - spot_list[1][1], 2))
                                if length < 5 : 
                                    spot_list.pop(-1)
                                    continue
                                
                                # spot_list[0] : width의 첫 번째 edge point
                                # spot_list[1] : width의 두 번째 edge point
                                # point : width의 두 edge point 사이의 midrid point
                                edge_point_list.append([spot_list[0], spot_list[1], point])
                                check_boolean = True
                                break
                            elif len(spot_list) > 2:
                                pass
                                
                        if not check_boolean: continue
                        else : break
            
            
            # calculate max length 
            max_length = 0
            max_length_idx = None
            for point_idx, edge_point in enumerate(edge_point_list):
                pt1_e, pt2_e , center_point = edge_point
                # use this code if you draw all each line between two edge point 
                # cv2.line(img, pt1_e, pt2_e, color=(255, 255, 0), thickness = 1)
                # cv2.circle(img, pt1_e, radius=2, color=(0, 255, 255), thickness=-1)
                # cv2.circle(img, pt2_e, radius=2, color=(0, 255, 255), thickness=-1)
                
                length_1 = math.sqrt(math.pow(pt1_e[0] - center_point[0], 2) + math.pow(pt1_e[1] - center_point[1], 2))     # midrid point와 width의 첫 번째 edge point 사이의 거리
                length_2 = math.sqrt(math.pow(center_point[0] - pt2_e[0], 2) + math.pow(center_point[1] - pt2_e[1], 2))     # midrid point와 width의 두 번째 edge point 사이의 거리

                if (length_1 > (length_2) * 5) or  ((length_1)* 5 < length_2) :  continue       # 두 거리의 차이가 5배 이상 나면 continue
                
                length = math.sqrt(math.pow(pt1_e[0] - pt2_e[0], 2) + math.pow(pt1_e[1] - pt2_e[1], 2))     # length가 가장 높은 것을 선택
                if max_length <= length: 
                    max_length = length
                    max_length_idx = point_idx

            if max_length_idx is not None:
                pt1_fe, pt2_fe, cross_point = edge_point_list[max_length_idx]
                onion_leaf_info["leaf_width_edge_coordinates"].append(pt1_fe)
                onion_leaf_info["leaf_width_edge_coordinates"].append(pt2_fe)
                
            else :  continue
            onion_leaf_info["midrid_point_coordinate"] = leaf_center_points
            onion_leaf_info["center"] = cross_point             

            point_dict_list.append(onion_leaf_info)   
            
        return point_dict_list
    
    
    def get_petiole_info(self):
        def get_midpoint_between_two_point_1(point_list):
            """
            point_list의 처음과 끝 점, 그리고 나머지 점들을 두 점씩 탐색하며
            단순히 두 점의 중점만을 append한다.
            """
            point_list_next_phase = []
            for i, point in enumerate(point_list):
                if i == 0 or i == len(point_list)-1: 
                    point_list_next_phase.append(point)
                    if i == len(point_list)-1: continue
                    
                point_list_next_phase.append([int((point[0] + point_list[i+1][0])/2), int((point[1] + point_list[i+1][1])/2)])
            return point_list_next_phase
    
    
        point_dict_list = []
            
        for idx in self.get_object_idx("petiole"):
            petiole_points = self.polygons[idx].copy()
        
            petiole_info = {}
            petiole_info["segmentation"] = petiole_points.copy()
            petiole_info["type"] = self.plant_type
            petiole_info["bbox"] = self.boxes[idx]

            # petiole의 boundary points를 계산 후 parsing_valiable_point을 통해 적절한 points들을 select
            dr = [[3, 2], [5, 3], [100, 2]]
            selected_point_x_sort = self.select_points(petiole_points, "width", dr)
            selected_point_y_sort = self.select_points(petiole_points, "height", dr)

            # selected_point_x_sort와 selected_point_y_sort를 사용하여 3단계의 parsing을 통해 최종 petiole_point_coordinate를 계산
            point_list_phase_1 = []
            for parsed_point in [selected_point_x_sort, selected_point_y_sort]:
                for point in parsed_point:
                    point_list_phase_1.append(point)
                    
            if self.return_width_or_height(self.boxes[idx]) == "height":
                point_list_phase_1.sort(key = lambda x: x[1], reverse=True)
            else:
                point_list_phase_1.sort()


            point_list_phase_2 = []
            before_point = None
            for i in range(len(point_list_phase_1)):
                if i == 0: 
                    point_list_phase_2.append(point_list_phase_1[0])
                    before_point = point_list_phase_1[0]

                min_length = 10000
                next_point = None
                
                for point_j in point_list_phase_1:
                    if point_j in point_list_phase_2: continue    
                    length = math.sqrt(math.pow(before_point[0] - point_j[0], 2) + math.pow(before_point[1] - point_j[1], 2))
                    if min_length>length:
                        min_length = length
                        next_point = point_j

                
                if next_point is not None:
                    point_list_phase_2.append(next_point)
                    before_point = next_point

            
            point_list_phase_3 = get_midpoint_between_two_point_1(point_list_phase_2)    # 두 개의 point에 대한 중점을 구하여 오차 감소
            point_list_phase_3 = get_midpoint_between_two_point_1(point_list_phase_3)
            point_list_phase_3 = self.select_valiable_point(point_list_phase_3, 100, 3)  # 2/3 만큼의 point를 제거

            petiole_info["petiole_point_coordinate"] = point_list_phase_3

            petiole_info["center"] = point_list_phase_3[len(point_list_phase_3)//2]


            point_dict_list.append(petiole_info)
        return point_dict_list


        
    # midrid_point를 한 번 솎아낸다.
    def parsing_onion_leaf_points(self, midrid_points):
        tmp_point_dict = []
        for idx in range(len(midrid_points)):
            if idx == 0 or idx == len(midrid_points)-1:
                tmp_point_dict.append(midrid_points[idx])
                continue
            
            x_before, y_before = midrid_points[idx-1][0], midrid_points[idx-1][1]
            x_this, y_this = midrid_points[idx][0], midrid_points[idx][1]
            x_next, y_next = midrid_points[idx+1][0], midrid_points[idx+1][1]
            
            slope_before, _ = get_slope_alpha(x_before, y_before, x_this, y_this)     # midrid의 현재 point와 이전 point사이의 기울기
            slope_next, _ = get_slope_alpha(x_this, y_this, x_next, y_next)         # midrid의 현재 point와 다음 point사이의 기울기
                
            if slope_before * slope_next < 0 :  # 두 기울기의 곱이 음수인 경우 현재 point의 위치를 재조정
                x_npoint, y_npoint = int((x_before + x_next)/2), int((y_before + y_next)/2)
                tmp_point_dict.append([x_npoint, y_npoint])
            elif (abs(slope_before) < 1 and abs(slope_next) > 1) or (abs(slope_before) > 1 and abs(slope_next) < 1):   # 기울기가 급격하게 변하는 경우 현재 point의 위치를 재조정
                x_npoint, y_npoint = int((x_before + x_next)/2), int((y_before + y_next)/2)
                tmp_point_dict.append([x_npoint, y_npoint])
            else: tmp_point_dict.append(midrid_points[idx])
        
        return tmp_point_dict

    def get_midpoint_between_two_point_2(self, point_list, distance, first_point = None, last_point = None, img = None):
        """
        point_list를 탐색하며 두 점 사이의 거리가 distance 보다 작을 시 그 중점을 append한다.
        """
        next_point_list = []
        
        append_flag = True
        for i, point_i in enumerate(point_list) :
            if first_point == True and i == 0:
                next_point_list.append(point_i)  
                continue 
            if last_point == True and i == len(point_list)-1 :
                next_point_list.append(point_i)  
                break
            

            minimum_point = None
            min_length = 10000
            
            for point_j in point_list[i+1:] :
                length = math.sqrt(math.pow(point_i[0] - point_j[0], 2) + math.pow(point_i[1] - point_j[1], 2))
                if min_length > length:
                    min_length = length
                    minimum_point = point_j
            
            # if min_length == 0.0: continue

            if minimum_point is not None:
                if min_length < distance:   # 10보다 작은 거리에 있는 점들은 그 중점을 append한다. 
                    next_point_list.append([int((point_i[0] + minimum_point[0])/2), int((point_i[1] + minimum_point[1])/2)])
                    append_flag = False
                else:
                    if append_flag : 
                        next_point_list.append(point_i)
                    append_flag = True
            
        return next_point_list


    
    def get_tomato_leaf_coordinates(self):
        """
            tomato의 leaf는 midrid를 기준으로 height와 width를 계산 
        """

        # sub function
        def except_points_in_bbox(bbox, point_list):
            x_min, y_min, x_max, y_max = bbox
            tmp_list = []
            for point in point_list:
                if x_min -10 < point[0] and point[0] < x_max + 10 and y_min - 10 < point[1] and point[1] < y_max + 10:
                    continue
                tmp_list.append(point)

            return tmp_list
        
        # sub function
        def get_midpoint_in_three_point(point_list, distance, first_point = None, last_point = None, img = None):
            """
            point_list를 순서대로 탐색하며 다음 점과의 거리, 다다음 점과의 거리가 크게 차이가 없을 때 세 점의 중점을 append한다.
            다음 점과의 거리는 distance를 shreshold로 한다.
            """
            next_point_list = []
            continue_count = 2
            for i, point_I in enumerate(point_list) :
                if first_point == True and i == 0:
                    next_point_list.append(point_I)  
                    continue 
                if last_point == True and i == len(point_list)-1 :
                    next_point_list.append(point_I)  
                    break

                if continue_count !=0 : 
                    continue_count -= 1
                    continue

                if last_point == True and i == len(point_list)-2 :
                    next_point_list.append(point_I)  
                    next_point_list.append(point_list[i+1])  
                    break
                elif i+1 >= len(point_list) - 1: 
                    next_point_list.append(point_I) 
                    break
                                
                
                point_II = point_list[i+1]
                point_III = point_list[i+2]

                length_II = math.sqrt(math.pow(point_I[0] - point_II[0], 2) + math.pow(point_I[1] - point_II[1], 2))
                length_III = math.sqrt(math.pow(point_I[0] - point_III[0], 2) + math.pow(point_I[1] - point_III[1], 2))
                if length_II < distance and length_II * 1.7 > length_III:
                    point = [int((point_I[0] + point_II[0] + point_III[0])/3), int((point_I[1] + point_II[1] + point_III[1])/3)]
                    next_point_list.append(point)
                    continue_count = 2
                else: 
                    next_point_list.append(point_I)
            return next_point_list
        
        # sub function
        def get_nearest_point(last_point, points_list):
            min_length = 100000

            for point in points_list : 
                length = math.sqrt(math.pow(last_point[0] - point[0], 2) + math.pow(last_point[1] - point[1], 2))
                if min_length > length : 
                    first_point = point
                    min_length = length
            
            return first_point
        
        # sub function
        def combind_two_list(previous_list, next_list, last_point = None, img = None):
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
            for _ in next_list : 
                min_length = 100000
                near_point = previous_list[-1]
                for point in next_list : 
                    if point in exist_points : continue            
                    
                    length = math.sqrt(math.pow(near_point[0] - point[0], 2) + math.pow(near_point[1] - point[1], 2))
                    if min_length > length :
                        min_length = length
                        nearest_point = point
                
        
                if last_point is not None:
                    if last_point == nearest_point :
                        exist_points.append(nearest_point)
                        previous_list.append(nearest_point)
                        break
                exist_points.append(nearest_point)
                previous_list.append(nearest_point)
                
            return previous_list

        # sub function
        def rearange_last_midrid_coordinates(last_midrid_coordinates):
            divided_last_midrid_coordinates_list = [last_midrid_coordinates[:int(len(last_midrid_coordinates)/3)], 
                                                    last_midrid_coordinates[int(len(last_midrid_coordinates)/3):int(len(last_midrid_coordinates)*2/3)],
                                                    last_midrid_coordinates[int(len(last_midrid_coordinates)*2/3):]]
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
                tmp_dict["center"] = [(x_max_tmp + x_min_tmp)//2, (y_max_tmp + y_min_tmp)//2]
                tmp_list.append(tmp_dict)                

            for i, dict_ in enumerate(tmp_list):
                if i == 0 or i == 1:
                    if dict_["width"] < dict_["height"]:
                        if tmp_list[i+1]["center"][1] > dict_["center"][1]: 
                            # 다음 y center가 더 높은 값이면 오름차순 y sort
                            divided_last_midrid_coordinates_list[i].sort(key = lambda x:x[1])  
                        else: 
                            divided_last_midrid_coordinates_list[i].sort(key = lambda x:x[1], reverse=True)
                    else:
                        if tmp_list[i+1]["center"][0] > dict_["center"][0] : 
                            # 다음 x center가 더 높은 값이면 오름차순 x sort
                            divided_last_midrid_coordinates_list[i].sort() 
                        else : 
                            divided_last_midrid_coordinates_list[i].sort(reverse = True)
                else:
                    if dict_["width"] < dict_["height"]:
                        if tmp_list[i-1]["center"][1] > dict_["center"][1]: 
                            # 이전 y center가 더 높은 값이면 내림차순 y sort
                            divided_last_midrid_coordinates_list[i].sort(key = lambda x:x[1], reverse=True)  
                        else: 
                            divided_last_midrid_coordinates_list[i].sort(key = lambda x:x[1])  
                    else:
                        if tmp_list[i-1]["center"][0] > dict_["center"][0] : 
                            # 이전 x center가 더 높은 값이면 내림차순 x sort
                            divided_last_midrid_coordinates_list[i].sort(reverse = True)
                        else : 
                            divided_last_midrid_coordinates_list[i].sort()

            last_midrid_coordinates = []
            for last_midrid_coordinates_part in divided_last_midrid_coordinates_list:
                last_midrid_coordinates += last_midrid_coordinates_part
            
            return last_midrid_coordinates
        
        # main function
        point_dict_list = []
        for outer_idx, inner_info_list in  self.outer_inner_idx['midrid'].items():
            bbox_midrid = self.boxes[int(outer_idx)]
            tomato_leaf_info = dict(segmentation = None,
                                    type = self.plant_type,
                                    bbox = bbox_midrid)    
            
            # first midrid 또는 last midrid중 하나라도 없는지 확인
            # first midrid 또는 last midrid중 하나라도 없는 경우는 continue
            first_midrid_flag, last_midrid_flag = False, False
            for inner_info in inner_info_list:
                _, inner_object_name = inner_info
                if inner_object_name == "first_midrid": first_midrid_flag = True
                elif inner_object_name == "last_midrid": last_midrid_flag = True
            if not(first_midrid_flag and last_midrid_flag): continue
            
            outer_idx = int(outer_idx)
            midrid_dict = dict()
            midrid_dict["midrid"] = dict(points = self.polygons[outer_idx].copy(),
                                                bbox = bbox_midrid)
            midrid_dict["mid_midrid"] = list()
                        
            for inner_info in inner_info_list:
                inner_idx, inner_object_name = inner_info
                
                if inner_object_name == "mid_midrid":
                    midrid_dict[inner_object_name].append(self.compute_center_point(self.boxes[inner_idx]))
                
                for object in ["first_midrid", "last_midrid"]:
                    if inner_object_name == object:        
                        midrid_dict[object] = dict(
                            points = self.polygons[inner_idx].copy(),
                            bbox = self.boxes[inner_idx])
            
            # midrid의 center points를 계산
            midrid_center_points = self.get_tomato_center_points(midrid_dict['midrid']["points"], 
                                                                 object = "midrid",
                                                                 width_or_height = None,
                                                                 mid_midrid_center_points = midrid_dict["mid_midrid"])
            # mid_midrid의 center points까지 append
            for center_x_y in midrid_dict["mid_midrid"]:
                midrid_center_points.append(center_x_y)
            
            
            ### phase_1
            # first_midrid가 midrid의 center보다 왼쪽에 있으면 midrid_center_points를 x 오름차순 sort
            # 그 반대는 x 내림차순 sort
            x_center_first, y_center_first = self.compute_center_point(midrid_dict["first_midrid"]["bbox"])
            x_center_midrid = self.compute_center_point(bbox_midrid)[0]
            if x_center_midrid > x_center_first: midrid_center_points.sort()   
            else : midrid_center_points.sort(reverse=True)
            midrid_points_phase_1 = midrid_center_points
            
            # combine first_midrid_coordinates and midmidrid_and_midrid_coordinates to point_list_phase_1
            # first_midrid영역 안에 있는 midmidrid_and_midrid_coordinates를 제외
            midrid_points_phase_1 = except_points_in_bbox(midrid_dict["first_midrid"]["bbox"], midrid_points_phase_1)
            
            # 가까이 모여있는 세 개의 point를 하나의 point로 치환
            midrid_points_phase_1 = get_midpoint_in_three_point(midrid_points_phase_1, 
                                                                10, 
                                                                first_point = True, 
                                                                last_point = False)    
            
            # get center coordinates of first_midrid and sort forward to right direction
            first_midrid_cneter_points = self.get_tomato_center_points(midrid_dict["first_midrid"]['points'], 
                                                                       object= "first",
                                                                       width_or_height= self.return_width_or_height(midrid_dict["first_midrid"]["bbox"]))        

            if self.return_width_or_height(midrid_dict["first_midrid"]["bbox"]) == "height":     # first_midrid의 세로길이가 클 때
                # first_midrid_coordinates의 y center coordinates가 midrid_points_phase_1 첫 번째 point의 y coordinate보다 낮은 곳에 위치한다면
                # first_midrid_coordinates를 y기준 오름차순 sort(image상으로는 아래가 높은 값)
                # 그 반대는 y기준 내림차순 sort
                if y_center_first < midrid_points_phase_1[0][1]:
                    first_midrid_cneter_points.sort(key = lambda x:x[1])
                else : first_midrid_cneter_points.sort(key = lambda x:x[1], reverse= True)
            else :                              # first_midrid의 가로길이가 클 때 
                if x_center_midrid > x_center_first: first_midrid_cneter_points.sort()
                else : first_midrid_cneter_points.sort(reverse=True)
                
            ### phase_2
            midrid_points_phase_2 = combind_two_list(first_midrid_cneter_points, midrid_points_phase_1)
           
            # get center coordinates of last_midrid and sort forward to right direction
            # last_midrid 에 대해서 x 또는 y sort
            last_midrid_coordinates = self.get_tomato_center_points(midrid_dict["last_midrid"]['points'], 
                                                                    object= "last",
                                                                    width_or_height= self.return_width_or_height(midrid_dict["last_midrid"]["bbox"]))    
            
            # last_midrid_영역 안에 있는 point_list_phase_2를 제외
            point_list_phase_2 = except_points_in_bbox(midrid_dict["last_midrid"]["bbox"], midrid_points_phase_2)
            if len(point_list_phase_2) == 0 : continue
            
            
            nearest_point_of_last_midrid = get_nearest_point(point_list_phase_2[-1], last_midrid_coordinates)
            
            x_center_last, y_center_last = self.compute_center_point(midrid_dict["last_midrid"]["bbox"])

            if self.return_width_or_height(midrid_dict["first_midrid"]["bbox"]) == "height":
                if y_center_last < nearest_point_of_last_midrid[1]:  
                    # midrid에 가장 가까운 point가 last midrid의 ceneter y coordinate보다 높으면(아래 위치) 내림차순 y sort
                    last_midrid_coordinates.sort(key = lambda x:x[1], reverse= True)
                else: 
                    last_midrid_coordinates.sort(key = lambda x:x[1])
            else:
                if x_center_last < nearest_point_of_last_midrid[0]:  
                    # midrid에 가장 가까운 point가 last midrid의 ceneter x coordinate보다 우측에 있으면 내림차순 x sort
                    last_midrid_coordinates.sort(reverse= True)
                else: 
                    last_midrid_coordinates.sort()
                        
            # last_midrid_coordinates를 3분할 후, 각각 y 또는 x sort를 진행한다.
            last_midrid_coordinates = rearange_last_midrid_coordinates(last_midrid_coordinates)     
            
            ### phase_3
            # combine last_midrid_coordinates and point_list_phase_2 to point_list_phase_3 
            point_list_phase_3 = combind_two_list(point_list_phase_2, last_midrid_coordinates, last_midrid_coordinates[-1])           
            first_point_of_phase_2 = [point_list_phase_2[0]]
            point_list_phase_3 = combind_two_list(first_point_of_phase_2, point_list_phase_3, point_list_phase_3[-1])

            ### phase_4
            point_list_phase_4 = get_midpoint_in_three_point(point_list_phase_3, 12, first_point = True, last_point = True ) 
            point_list_phase_4 = self.get_midpoint_between_two_point_2(point_list_phase_4, 10, first_point = True, last_point = True)

            tomato_leaf_info["midrid_point_coordinate"] = point_list_phase_4    
            tomato_leaf_info["center"] = point_list_phase_4[len(point_list_phase_4)//2]

            # ### width 
            # if len(total_midrid_idx) == 5:
            #     side_midrid_coordinates_list = []
            #     side_midrid_info_list = []
            #     # side_midrid_meta : [idx, side_midrid_mask_coordinates]
            #     for side_midrid_idx_coordinate in tomato_meta_list_dict["side_midrid_meta_list"] :
            #         # total_midrid_idx[4] : side_midrid_meta_list
            #         # side_midrid_meta_list : [[idx, iou_side_part_bbox, side_part_bbox_list], ...]
            #         for side_midrid_meta in total_midrid_idx[4]:
                        
                        
            #             if side_midrid_idx_coordinate[0] == side_midrid_meta[0]:
                            
            #                 side_midrid_coordinates_list.append(side_midrid_idx_coordinate[1])
            #                 # side_midrid_info_list : [[iou_side_part_bbox, side_part_bbox_list, side_midrid_bbox]]
            #                 side_midrid_info_list.append([side_midrid_meta[1], side_midrid_meta[2], self.boxes[side_midrid_meta[0]]])                               
                
            #     # print(f"len(side_midrid_coordinates_list) : {len(side_midrid_coordinates_list)}")
                
            #     left_max_length, right_max_length = -1, -1
            #     left_width_point, right_width_point = [], []
            #     left_ot_rigth = None
            #     for side_midrid_coordinates, side_midrid_info in zip(side_midrid_coordinates_list, side_midrid_info_list):
            #         width_points_list = []

            #         side_midrid_iou_bbox = side_midrid_info[0]    # side midrid를 10개로 분한할 것들 중 mid_midrid와의 iou값이 0이 아닌 bbox의 좌표
            #         side_part_bbox_list = side_midrid_info[1]   # side midrid를 10개로 분할한 것의 각 bbox 좌표들

            #         for j, side_midrid_bbox in enumerate(side_part_bbox_list):
            #             if side_midrid_iou_bbox == side_midrid_bbox and j < len(side_part_bbox_list)/3:     # iou가 0이 아닌 bbox가 side midrid의 초반 part에 등장할 때
            #                                                                         # side_midrid가 leaf기준 오른쪽(leaf가 수직방향인 경우) 또는 leaf기준 아래(leaf가 수평방향일 경우)에 위치한다.
            #                 x_min, y_min, x_max, y_max = side_midrid_iou_bbox
            #                 width_points_list.append([(x_min + x_max)//2, (y_min + y_max)//2])    # side midrid를 10개로 분한할 것들 중 mid_midrid와의 iou값이 0이 아닌 bbox의 center point
            #                 left_ot_rigth = "left"
            #                 width_points_list.append(side_midrid_coordinates[-1])  # 첫 번쨰 point
            #             elif side_midrid_iou_bbox == side_midrid_bbox and j > len(side_part_bbox_list)*2/3:   # iou가 0이 아닌 bbox가 side midrid의 후반 part에 등장할 때
            #                                                                         # side_midrid leaf기준 왼쪽(leaf가 수직방향인 경우) 또는 leaf기준 위(leaf가 수평방향일 경우)에 위치한다.
            #                 x_min, y_min, x_max, y_max = side_midrid_iou_bbox
            #                 width_points_list.append([(x_min + x_max)//2, (y_min + y_max)//2]) 
            #                 left_ot_rigth = "right"
            #                 width_points_list.append(side_midrid_coordinates[0])  # 첫 번쨰 point
            #             else:
            #                 x_min, y_min, x_max, y_max = side_midrid_bbox
            #                 width_points_list.append([(x_min + x_max)//2, (y_min + y_max)//2]) 
                    
            #         if left_ot_rigth == None : continue

                        
            #         # sort
            #         side_midrid_x_min, side_midrid_y_min, side_midrid_x_max, side_midrid_y_max = side_midrid_info[2]
            #         if side_midrid_x_max - side_midrid_x_min < side_midrid_y_max - side_midrid_y_min :  width_points_list.sort(key = lambda x : x[1])
            #         else: width_points_list.sort()
                    
            #         # 각 point간의 length를 합산  
            #         sum_length = 0
            #         for i, point in enumerate(width_points_list):
            #             if i == 0:
            #                 before_point = point
            #                 continue
            #             length = math.sqrt(math.pow(point[0] - before_point[0], 2) + math.pow(point[1] - before_point[1], 2))
            #             sum_length+= length

            #         if left_ot_rigth == "left" and sum_length > left_max_length:
            #             left_max_length = sum_length
            #             left_width_point = width_points_list
            #         elif left_ot_rigth == "right" and sum_length > right_max_length:
            #             right_max_length = sum_length
            #             right_width_point = width_points_list

 
            #     self.point_dict["leaf_width_edge_coordinates"] = dict()
            #     self.point_dict["leaf_width_edge_coordinates"]["left"] = left_width_point
            #     self.point_dict["leaf_width_edge_coordinates"]["right"] = right_width_point  
                    
            #     for point in left_width_point:
            #         cv2.circle(img, point, thickness=-1, radius=10, color = (255, 0, 0))
            #     for point in right_width_point:
            #         cv2.circle(img, point, thickness=-1, radius=10, color = (0, 0, 255))

                        
            #     for i, point in enumerate(point_list_phase_4) : 
            #         if i == 0: 
            #             cv2.circle(img, point, thickness=-1, radius=3, color= (255, 0, 0))
            #             before_point = point
            #             continue
                        
            #         cv2.circle(img, point, thickness=-1, radius=3, color= (255, 0, 0))
            #         cv2.line(img, before_point, point, thickness=2, color= ())
            #         before_point = point

            # else:
            #     pass
            
            # point_dict_list.append(self.point_dict)

            # # point_dict_list[0].keys() : ["midrid_point_coordinate", "leaf_width_edge_coordinates"]
            # #   point_dict_list[0]["midrid_point_coordinate"] : list
            # #   point_dict_list[0]["leaf_width_edge_coordinates"]["left"] : list
            # #   point_dict_list[0]["leaf_width_edge_coordinates"]["right"] : list
            point_dict_list.append(tomato_leaf_info)
        
        return point_dict_list
    

    def get_stem_first_leaf_dix_list(self, main_bbox_list, sub_center_point_list, main_object_idx_list, sub_object_idx_list, img):
        """               
        main_bbox_list : [bbox_1. bbox_2, ..., bbox_N]
            bbox = main_object의 bbox : [x_min, y_min, x_max, y_max]
        sub_center_point_list : [center_point_1, center_point_2, ..., center_point_N]
            center_point : sub_object의 중앙 좌표  : [x_center, y_center] 
        """
        main_sub_idx_list = []       # 한 개의 main과 그에 포함되는 sub에 대한 index를 list로 저장
        for i, bbox in enumerate(main_bbox_list):
            x_min, y_min, x_max, y_max = bbox
            sub_list = []
                        
            for j, center_point in enumerate(sub_center_point_list):
                x_center, y_center = center_point
                if x_min < x_center and x_center < x_max and y_min < y_center and y_max > y_center:
                    # j번째 sub의 center cooridnate가 i번째 main의 Bbox안에 포함될 경우, j번째 sub는 i번째 main의 sub_object이다. 
                    sub_list.append(sub_object_idx_list[j])

                    # main_object와 sub_object중 하나라도 대칭되는 것이 없으면 사용 안하는 mask
                    self.useful_mask_idx.append(sub_object_idx_list[j])         
                    
            main_sub_idx_list.append([main_object_idx_list[i], sub_list])
            self.useful_mask_idx.append(main_object_idx_list[i])

        # len(main_sub_idx_list) == sub_object를 포함한 main_object의 개수
        # main_sub_idx_list : [[idx of main_object_1, idx_list of sub_object_1], [idx of main_object_2, idx_list of sub_object_2], ...]
        # idx_list of sub_object_1 : [idx of sub_object_1, idx of sub_object_2, ...]
        return main_sub_idx_list 


    def get_midrid_leaf_idx_list(self, main_bbox_list, sub_center_point_list, main_object_idx_list, sub_object_idx_list):
        """               
        main_bbox_list : [bbox_1. bbox_2, ..., bbox_N]
            bbox = main_object의 bbox : [x_min, y_min, x_max, y_max]
        sub_center_point_list : [center_point_1, center_point_2, ..., center_point_N]
            center_point : sub_object의 중앙 좌표  : [x_center, y_center] 
        """
                  
        main_sub_idx_list = []       # 한 개의 main과 그에 포함되는 sub에 대한 index를 list로 저장
        for i, bbox in enumerate(main_bbox_list):
            x_min, y_min, x_max, y_max = bbox
            for j, center_point in enumerate(sub_center_point_list):
                x_center, y_center = center_point
                if x_min < x_center and x_center < x_max and y_min < y_center and y_max > y_center:
                    # j번째 sub의 center cooridnate가 i번째 main의 Bbox안에 포함될 경우, j번째 sub는 i번째 main의 sub_object이다. 
                        
                    main_sub_idx_list.append([sub_object_idx_list[j], main_object_idx_list[i]])  # 전체 object중 [midrid의 idx, leaf의 idx] 

                    # main_object와 sub_object중 하나라도 대칭되는 것이 없으면 사용 안하는 mask
                    self.useful_mask_idx.append(sub_object_idx_list[j])
                    self.useful_mask_idx.append(main_object_idx_list[i])
        # len(main_sub_idx_list) == sub_object를 포함한 main_object의 개수
        # main_sub_idx_list : [[idx of sub_object_1, idx of main_object_1], [idx of sub_object_2, idx of main_object_2], ...]
        return main_sub_idx_list   


# TODO : for test
    def get_idx_list(self, bbox_list, midrid_box, set_which):
        # bbox_list : mid_midrid, first_midrid, last_midrid 중 한 개의 bbox들의 list 
        x_min_midrid, y_min_midrid, x_max_midrid, y_max_midrid = midrid_box

        idx_list = []
        for j, box in enumerate(bbox_list):
            # bbox정보로부터 center coordinate를 계산
            x_center, y_center = (box[0] + box[2])/2, (box[1] + box[3])/2
            

            # center coordinate가 main midrid의 bbox영역 안에 위치한다면 idx_list에 append
            if (x_min_midrid < x_center and x_center < x_max_midrid
                and y_min_midrid < y_center and y_center < y_max_midrid) :

                if set_which == "mid": 
                    idx_list.append([self.mask_idx_dict["mid_midrid"][j], [x_center, y_center]])
                    
                    self.useful_mask_idx.append(self.mask_idx_dict["mid_midrid"][j])
                    
                elif set_which == "first" : idx_list.append([self.mask_idx_dict["first_midrid"][j], [x_center, y_center]])
                
                elif set_which == "last" : idx_list.append([self.mask_idx_dict["last_midrid"][j], [x_center, y_center]])
                
        
        return idx_list


    def get_real_midrid(self, midrid_list, mid_midrid_idx_list, midrid_box):
        if len(midrid_list) > 1 :
            tmp_point = [0, 0]
            for mid_midrid in mid_midrid_idx_list:
                tmp_point[0] += mid_midrid[1][0]
                tmp_point[1] += mid_midrid[1][1]

            if len(mid_midrid_idx_list) == 0:
                x_center, y_center = (midrid_box[0] + midrid_box[2])/2, (midrid_box[1] + midrid_box[3])/2
                mean_midrid_center_point = [x_center, y_center]
            else:
                mean_midrid_center_point = [tmp_point[0]/len(mid_midrid_idx_list), tmp_point[1]/len(mid_midrid_idx_list)]
            
            min_length = 100000
            for bbox_midrid in midrid_list:
                bbox_x_center, bbox_y_center = bbox_midrid[1]

                # mid_midrid 들의 center coordinate의 mean값 에 가장 가까운 first_midrid가 여러 first_midrid중 진짜 (len(mid_midrid_idx_list) != 0 일 때만. 
                # 그 외는 midrid의 center에서 먼 것을 선택)
                length = math.sqrt(math.pow(mean_midrid_center_point[0] - bbox_x_center, 2) + math.pow(mean_midrid_center_point[1] - bbox_y_center, 2))

                if min_length > length:
                    min_length = length
                    real_first_midrid_idx = bbox_midrid[0]
            
        elif len(midrid_list) == 1 : real_first_midrid_idx = midrid_list[0][0]
        else : real_first_midrid_idx = None

        # real_first_bbox_midrid : [idx, [x_center, y_center]
        
        if real_first_midrid_idx is not None:
            self.useful_mask_idx.append(real_first_midrid_idx)

        return real_first_midrid_idx



    def get_midrid_idx_list_for_test(self): 
        total_midrid_idx_list = []
        
                            
                               
        
        
        # self.tomato_bbox_list_dict["bbox_midrid_list"] : 1개의 image안에 있는 midrid들의 list 
        for i, midrid_box in enumerate(self.tomato_bbox_dict["midrid"]):   
            self.useful_mask_idx.append(self.mask_idx_dict["midrid"][i])
            
            
            # mid_midrid_idx_list : [[idx, [x_center, y_center]], [idx, [x_center, y_center]], ...]
            mid_midrid_idx_list = self.get_idx_list(self.tomato_bbox_dict["mid_midrid"], midrid_box, "mid")               
            
            # first_midrid_idx : [idx, [x_center, y_center]
            first_midrid_idx_list = self.get_idx_list(self.tomato_bbox_dict["first_midrid"], midrid_box, "first")
            first_midrid_idx = self.get_real_midrid(first_midrid_idx_list, mid_midrid_idx_list, midrid_box)

            # last_midrid_idx : [idx, [x_center, y_center]
            last_midrid_idx_list = self.get_idx_list(self.tomato_bbox_dict["last_midrid"], midrid_box, "last")
            last_midrid_idx = self.get_real_midrid(last_midrid_idx_list, mid_midrid_idx_list, midrid_box)

            only_mid_midrid_idx_list = []
            for mid_midrid_idx in mid_midrid_idx_list:
                only_mid_midrid_idx_list.append(mid_midrid_idx[0])

            # total_midrid_idx_list 의 각 원소 안에 mid_midrd, first_midrid, last_midrid에 대한 idx를 할당한다.
            # total_midrid_idx_list[0] : [midrid_idx, first_midrid_idx, mid_midrid_idx_list, last_midrid_idx]
            # mid_midrid_idx_list : [idx_1, idx_2, ..., idx_n]
            total_midrid_idx_list.append([self.mask_idx_dict["midrid"][i], first_midrid_idx, only_mid_midrid_idx_list, last_midrid_idx])
            
        
        return total_midrid_idx_list




    def x_or_y_center_coordinates(self, x_coordinates, y_coordinates, width_or_height):
        """
            coordinates : x좌표 list -> other_coordinates : y좌표 list
            coordinates : y좌표 list -> other_coordinates : x좌표 list
        """

        coordinates_no_dupl = list(set(y_coordinates)) if width_or_height == "height" else list(set(x_coordinates))

        temp_list_1 = []
        temp_list_2 = []
        if width_or_height == "height":
            for no_dulp in coordinates_no_dupl:     # slicing
                for i in range(len(y_coordinates)):   
                    if no_dulp == y_coordinates[i]:                      
                        temp_list_1.append([x_coordinates[i], no_dulp]) 
            # temp_list_1[N][1] : y coordinates of segmentation boundary (removed redundant elements)
            # temp_list_1[N][0] : x coordinates of segmentation boundary
            # temp_list_1[N][0]는 temp_list_1[M][0]과 중복되는 경우가 있다. (M != N인 임의의 수)
            
            ## remove redundant x coordinates of segmentation boundary
            for temp_coordi in temp_list_1:
                if temp_coordi not in temp_list_2:
                    temp_list_2.append(temp_coordi)
                else:
                    continue
            
            temp_list_2.sort(key=lambda x: x[1])  	# sort according to y coordinates
            # temp_list_2[N][1] : y coordinates of segmentation boundary (removed redundant elements)
            # temp_list_2[N][0] : x coordinates of segmentation boundary (removed redundant elements)
            # 각각의 y coordiante 에 대응되는 x coodinate는 1개 또는 2개 이상이다.

        else : 
            for no_dulp in coordinates_no_dupl:     # slicing
                for i in range(len(x_coordinates)):   
                    if no_dulp == x_coordinates[i]:                      
                        temp_list_1.append([no_dulp, y_coordinates[i]]) 
            
            for temp_coordi in temp_list_1:
                if temp_coordi not in temp_list_2:
                    temp_list_2.append(temp_coordi)
                else:
                    continue

            temp_list_2.sort()	


        return temp_list_2


    def get_sorted_center_points(self, points, width_or_height = None, img = None):
        """
        calculate midird coordinates and return a list of midird coordinates
        """

        
        x_coordinates, y_coordinates = [], []
        for point in points:
            x_coordinates.append(point[0])
            y_coordinates.append(point[1])
      
       
        sorted_coordinates = self.x_or_y_center_coordinates(x_coordinates, y_coordinates, width_or_height)    

        midrid_center_coordinates = []
        list_for_check_redundant = []
        total_other_value = None
        count = None

        for idx_, point in enumerate(sorted_coordinates):
            if width_or_height == "height":
                if point[1] not in list_for_check_redundant:      
                    # 이미 check한 y좌표가 아니라면
                    list_for_check_redundant.append(point[1])

                    if idx_ == 0 : 
                        total_other_value = point[0]
                        count = 1
                        continue

                    if count == 1:      #   y coordiante 에 대응되는 x coodinate가 1개인 경우
                        midrid_center_coordinates.append(point)

                    else:               #   y coordiante 에 대응되는 x coodinate가 2개 이상인 경우
                        midrid_center_coordinates.append([int(total_other_value / count), point[1]])
                        # int(total_x_value / count) == mean of x coordinate
                        # center_coor_no_dupl[1]  == y coordinate

                    total_other_value = point[0]
                    count = 1
                else :      
                    # 이미 check한 y좌표라면, x좌표 값을 add
                    total_other_value += point[0]
                    count +=1
            else:
                if point[0] not in list_for_check_redundant:      
                    # 이미 check한 y좌표가 아니라면
                    list_for_check_redundant.append(point[0])

                    if idx_ == 0 : 
                        total_other_value = point[1]
                        count = 1
                        continue

                    if count == 1:      #   y coordiante 에 대응되는 x coodinate가 1개인 경우
                        midrid_center_coordinates.append(point)

                    else:               #   y coordiante 에 대응되는 x coodinate가 2개 이상인 경우
                        midrid_center_coordinates.append([point[0], int(total_other_value / count)])
                        # int(total_x_value / count) == mean of x coordinate
                        # center_coor_no_dupl[1]  == y coordinate

                    total_other_value = point[1]
                    count = 1
                else :      
                    # 이미 check한 y좌표라면, x좌표 값을 add
                    total_other_value += point[1]
                    count +=1
            
        if width_or_height == "height": midrid_center_coordinates.sort(key = lambda x:x[1])
        else: midrid_center_coordinates.sort()

        return midrid_center_coordinates

    def get_draw_leaf_info(self, img = None):
        """
        leaf의 boundary points와 midrid points, type, width points 등을 계산
     
        Return
            point_dict_list : [point_dict, point_dict, point_dict...]
                point_dict.keys() = ["midrid_point_coordinate", "leaf_width_edge_coordinates"]
                point_dict["midrid_point_coordinate"] : [point_1, point_2, ... point_n]
                point_dict["leaf_width_edge_coordinates"] : [edge_right, edge_left]   or  [edge_left, edge_right] 
        """       
        
        point_dict_list = []        # 각 leaf에 대해서 midrid의 points의 좌표, leaf의 폭 꼭지점의 좌표를 담게 된다.
                                    # [point_dict, point_dict, point_dict...]

                                  
        for outer_idx, inner_info_list in  self.outer_inner_idx['leaf'].items():
            outer_idx = int(outer_idx)
            
            if len(inner_info_list) == 0: continue      # midrid가 없는 lead일 경우 comtinue
            
            inner_idx, _ = inner_info_list[0]       # midrid는 각 leaf당 1개만 할당되어있다고 가정
            leaf_points = self.polygons[outer_idx].copy()
            midrid_points = self.polygons[inner_idx].copy()
            bbox_midrid = self.boxes[inner_idx]
            bbox_leaf = self.boxes[outer_idx]
            more_longer = self.return_width_or_height(bbox_midrid)
            midrid_center_points = self.get_sorted_center_points(midrid_points, 
                                                                 width_or_height = more_longer)

            if len(midrid_center_points) == 0: continue
      
            if more_longer:
                midrid_center_points.sort(key = lambda x:x[1])  
            else:
                midrid_center_points.sort()
                

            self.point_dict = dict(midrid_point_coordinate = [],            # [point_1, point_2, ....]
                                   leaf_width_edge_coordinates = [],        # [edge_right, edge_left]   or  [edge_left, edge_right] 
                                   segmentation = leaf_points.copy(),
                                   type = self.plant_type,
                                   bbox = bbox_leaf)

            # midrid의 point개수 설정
            num_spot = self.config.NUM_SPOT       # c_cfg.NUM_SPOT == 10 이면 10개의 point를 찍는다.
            if self.plant_type == "strawberry":
                num_spot = int(num_spot/2)
            if self.plant_type in ["chilipepper_seed", "cucumber_seed", "chili"]:
                num_spot = int(num_spot/3)

            count_bot = 1
            ### select particular point
            # midrid point 확보
            for i, center_coordinate in enumerate(midrid_center_points):
                if i == 0 or\
                   i == int(len(midrid_center_points)*(count_bot/num_spot)) or\
                   i == (len(midrid_center_points) -1):    
                    count_bot +=1
                    self.point_dict["midrid_point_coordinate"].append(center_coordinate)

            
            if self.plant_type in ["paprika", "strawberry", "chilipepper_seed", "cucumber_seed"]:
                # midrid가 leaf의 끝에 닿지 않는 경우 midrid의 point를 연장해서 그린다.
                self.find_first_point_midrid(leaf_points, more_longer)
                self.find_last_point_midrid(leaf_points, more_longer)
                # self.point_dict["midrid_point_coordinate"] = self.get_midpoint_between_two_point_2(self.point_dict["midrid_point_coordinate"], 5 , first_point = False, last_point = False)

            elif self.plant_type in ["melon" ,"cucumber"]:
                # center point에 더 가까이 있는 midrid가 first midrid point임을 설정한 후 last point를 연장할지, first point를 연장할지 결정
                center_point = int((bbox_leaf[0] + bbox_leaf[2])/2), int((bbox_leaf[1] + bbox_leaf[3]) /2) 
                x_first, y_first = self.point_dict["midrid_point_coordinate"][0][0], self.point_dict["midrid_point_coordinate"][0][1]
                x_last, y_last = self.point_dict["midrid_point_coordinate"][-1][0], self.point_dict["midrid_point_coordinate"][-1][1]

                length_from_first_to_center = math.sqrt(math.pow(center_point[0] - x_first, 2) + math.pow(center_point[1] - y_first, 2))
                length_from_last_to_center = math.sqrt(math.pow(center_point[0] - x_last, 2) + math.pow(center_point[1] - y_last, 2))
                
                # cv2.circle(img, (x_first, y_first), thickness=-1, radius=10, color=(255, 255, 0))
                # cv2.circle(img, (x_last, y_last), thickness=-1, radius=10, color=(255, 0, 255))
                # cv2.circle(img, center_point, thickness=-1, radius=10, color=(0, 0, 255))
                if length_from_first_to_center < length_from_last_to_center : # first가 center에 더 가까이 있을 경우
                    self.find_last_point_midrid(leaf_points, more_longer)
                else : # last가 center에 더 가까이 있을 경우
                    self.find_first_point_midrid(leaf_points, more_longer)

            
            # calculate edge point about width of leaf base on point_coordinate
            check_availability, cross_point = self.find_width_point_midrid(leaf_points, bbox_leaf, more_longer, img = img)

            if len(self.point_dict["leaf_width_edge_coordinates"]) == 0:
                continue
            self.point_dict["leaf_width_edge_coordinates"] = get_width_point(self.point_dict["leaf_width_edge_coordinates"], 5)    
            
            if cross_point is not None:
                self.point_dict["center"] = cross_point
            

            # midrid의 coordinates가 작은 영역에 뭉쳐있다면 point_dict_list에 append하지않음
            first_point, last_point = self.point_dict["midrid_point_coordinate"][0], self.point_dict["midrid_point_coordinate"][-1]
            length = math.sqrt(math.pow(first_point[0] - last_point[0], 2) + math.pow(first_point[1] - last_point[1], 2))
            
            box_width, box_height = self.compute_width_height(bbox_midrid)
            if more_longer == "width":
                if length < box_width/7:
                    check_availability = False
            elif more_longer == "height":
                if length < box_height/7:
                    check_availability = False

            if check_availability:
                point_dict_list.append(self.point_dict)      

        return point_dict_list


    def find_first_point_midrid(self, leaf_coordinates, width_or_height):

        x_first, y_first = self.point_dict["midrid_point_coordinate"][0][0], self.point_dict["midrid_point_coordinate"][0][1]
        x_second, y_second = self.point_dict["midrid_point_coordinate"][1][0], self.point_dict["midrid_point_coordinate"][1][1]

        slope, alpha = get_slope_alpha(x_second, y_second, x_first, y_first)

        continue_check, continue_num, edge_list, skip_num = create_check_flag(leaf_coordinates)                                                                               

        check_boolean = False
        for i in range(self.margin_error):
            for x, y in leaf_coordinates:   # leaf의 coordinate를 탐색하며 y = slope*x + alpha 에 만족하는 y, x coordinate를 찾는다.
                edge_list, continue_num, continue_check = self.find_coordinate_slicing(continue_check, x, y, slope, alpha, edge_list, continue_num, skip_num, False, i)  
                if len(edge_list) == 2:
                    length_1 = math.sqrt(math.pow(edge_list[0][0] - x_first, 2) + math.pow(edge_list[0][1] - y_first, 2))
                    length_2 = math.sqrt(math.pow(edge_list[1][0] - x_first, 2) + math.pow(edge_list[1][1] - y_first, 2))
                    check_boolean = True                 

                    if abs(length_1 - length_2) < (length_1 + length_2)/10:
                        edge_list.pop(-1)
                        continue
                    
                    # 두 좌표 중 더 가까운 곳의 좌표를 얻는다.
                    if length_1 <= length_2:
                        self.insert_to_point_dict(edge_list, width_or_height, 0)    # 긴 line이 length_2일 때, 짧은 부분의 point가 가까운 point
                    else: 
                        self.insert_to_point_dict(edge_list, width_or_height, 1)      # first point가 우측에 있고, 긴 line이 length_1이다.
                    break
            if not check_boolean: continue
            else : break
            

    def insert_to_point_dict(self, edge_list, width_or_height, num):
        if width_or_height =="width":
            # 이미 midrid의 첫 번째 point가 새롭게 찾은 edge보다 x값이 더 작은 경우는 pass
            if self.point_dict["midrid_point_coordinate"][-1][0] <= edge_list[num][0]:
                pass
            else :
                self.point_dict["midrid_point_coordinate"].insert(0, [edge_list[num][0], edge_list[num][1]])
        else:
            # 이미 midrid의 첫 번째 point가 새롭게 찾은 edge보다 y값이 더 작은 경우는 pass
            if self.point_dict["midrid_point_coordinate"][-1][1] <= edge_list[num][1]:
                pass
            else :
                self.point_dict["midrid_point_coordinate"].insert(0, [edge_list[num][0], edge_list[num][1]])
        

    def find_last_point_midrid(self, leaf_coordinates, width_or_height):
        x_second_last, y_second_last = self.point_dict["midrid_point_coordinate"][-2][0], self.point_dict["midrid_point_coordinate"][-2][1]
        x_last, y_last = self.point_dict["midrid_point_coordinate"][-1][0], self.point_dict["midrid_point_coordinate"][-1][1]
        
        slope, alpha = get_slope_alpha(x_second_last, y_second_last, x_last, y_last)
        # y_last = slope*x_last + alpha : (x_last, y_last)에서 기울기 slope를 가진 1차함수
        
        continue_check, continue_num, edge_list, skip_num = create_check_flag(leaf_coordinates)   

        check_boolean = False
        for i in range(self.margin_error):
            for x, y in leaf_coordinates:   # leaf의 coordinate를 탐색하며 y = slope*x + alpha 에 만족하는 y, x coordinate를 찾는다.
                edge_list, continue_num, continue_check = self.find_coordinate_slicing(continue_check, x, y, slope, alpha, edge_list, continue_num, skip_num, False, i)
                if len(edge_list) == 2:
                    length_1 = math.sqrt(math.pow(edge_list[0][0] - x_last, 2) + math.pow(edge_list[0][1] - y_last, 2))
                    length_2 = math.sqrt(math.pow(edge_list[1][0] - x_last, 2) + math.pow(edge_list[1][1] - y_last, 2))
                    
                    if abs(length_1 - length_2) < (length_1 + length_2) /3 :    # 두 좌표가 모두 기본적으로 x_last, y_last와 멀 경우는 예외
                        edge_list.pop(-1)
                        break 
                    
                    check_boolean = True
                    if length_1 <= length_2:
                        self.append_to_leaf_point_dict(edge_list, width_or_height, 0)
                    else: 
                        self.append_to_leaf_point_dict(edge_list, width_or_height, 1)
                    break        
            if not check_boolean: continue
            else : break
            


    def append_to_leaf_point_dict(self, edge_list, width_or_height, num):
        if width_or_height =="width":
            # 이미 midrid의 마지막 point가 새롭게 찾은 edge보다 x값이 더 큰 경우는 pass
            if self.point_dict["midrid_point_coordinate"][-1][0] >= edge_list[num][0]:
                pass
            else :
                self.point_dict["midrid_point_coordinate"].append([edge_list[num][0], edge_list[num][1]])
        else:
            # 이미 midrid의 마지막 point가 새롭게 찾은 edge보다 y값이 더 큰 경우는 pass
            if self.point_dict["midrid_point_coordinate"][-1][1] >= edge_list[num][1]:
                pass
            else :
                self.point_dict["midrid_point_coordinate"].append([edge_list[num][0], edge_list[num][1]])
        

    def find_width_point_midrid(self, leaf_coordinates, bbox_leaf, width_or_height, img = None):
        # calculate center box coordinates
        x_min, y_min, x_max, y_max = bbox_leaf
        width, height = x_max - x_min, y_max - y_min   

        if width_or_height == "height" : 
            if self.plant_type in ["melon", "cucumber", "strawberry"] : 
                x_min_center_box, x_max_center_box, y_min_center_box, y_max_center_box = x_min, x_max, y_min, int(y_max - height/4)
            elif self.plant_type in ["paprika", "chilipepper_seed", "cucumber_seed", "chili"] : 
                x_min_center_box, x_max_center_box, y_min_center_box, y_max_center_box = x_min, x_max, y_min, int(y_max - height/3)
            
        else :
            if self.plant_type in ["melon", "cucumber", "strawberry"] :
                x_min_center_box, x_max_center_box, y_min_center_box, y_max_center_box = int(x_min + width/10), int(x_max - width/4), y_min, y_max
            elif self.plant_type in ["paprika", "chilipepper_seed", "cucumber_seed", "chili"] : 
                x_min_center_box, x_max_center_box, y_min_center_box, y_max_center_box = x_min, x_max, y_min, y_max
        
        # -----TODO : for 박람회        # 2022-12-12
        # x_first, y_first = self.point_dict["midrid_point_coordinate"][0][0], self.point_dict["midrid_point_coordinate"][0][1]
        # x_last, y_last = self.point_dict["midrid_point_coordinate"][-1][0], self.point_dict["midrid_point_coordinate"][-1][1]
        # midrid_slope, _ = get_slope_alpha(x_first, y_first, x_last, y_last)        # midrid의 첫 번째 point와 마지막 point사이의 기울기 
        
        # midrid_point를 한 번 솎아낸다.
        tmp_point_dict = []
        for idx in range(len(self.point_dict["midrid_point_coordinate"])):
            if idx == 0 or idx == len(self.point_dict["midrid_point_coordinate"])-1:
                tmp_point_dict.append(self.point_dict["midrid_point_coordinate"][idx])
                continue
            
            x_before, y_before = self.point_dict["midrid_point_coordinate"][idx-1][0], self.point_dict["midrid_point_coordinate"][idx-1][1]
            x_this, y_this = self.point_dict["midrid_point_coordinate"][idx][0], self.point_dict["midrid_point_coordinate"][idx][1]
            x_next, y_next = self.point_dict["midrid_point_coordinate"][idx+1][0], self.point_dict["midrid_point_coordinate"][idx+1][1]
            
            slope_before, _ = get_slope_alpha(x_before, y_before, x_this, y_this)     # midrid의 현재 point와 이전 point사이의 기울기
            slope_next, _ = get_slope_alpha(x_this, y_this, x_next, y_next)         # midrid의 현재 point와 다음 point사이의 기울기
                
            if slope_before * slope_next < 0 :  # 두 기울기의 곱이 음수인 경우 현재 point의 위치를 재조정
                x_npoint, y_npoint = int((x_before + x_next)/2), int((y_before + y_next)/2)
                tmp_point_dict.append([x_npoint, y_npoint])
            elif (abs(slope_before) < 1 and abs(slope_next) > 1) or (abs(slope_before) > 1 and abs(slope_next) < 1):   # 기울기가 급격하게 변하는 경우 현재 point의 위치를 재조정
                x_npoint, y_npoint = int((x_before + x_next)/2), int((y_before + y_next)/2)
                tmp_point_dict.append([x_npoint, y_npoint])
            else: tmp_point_dict.append(self.point_dict["midrid_point_coordinate"][idx])

        self.point_dict["midrid_point_coordinate"] = tmp_point_dict
        
        
        edge_point_list = []
        for idx in range(len(self.point_dict["midrid_point_coordinate"])):
            if idx == 0 or idx == len(self.point_dict["midrid_point_coordinate"])-1 : continue

            x_this, y_this = self.point_dict["midrid_point_coordinate"][idx][0], self.point_dict["midrid_point_coordinate"][idx][1]
            x_next, y_next = self.point_dict["midrid_point_coordinate"][idx+1][0], self.point_dict["midrid_point_coordinate"][idx+1][1]
            
            if not(x_this > x_min_center_box and x_this < x_max_center_box \
                and y_this > y_min_center_box and y_this < y_max_center_box) : continue  # 현재 점이 center box안에 위치하지 않는 경우 
            slope, _ = get_slope_alpha(x_this, y_this, x_next, y_next)
            
            # -----TODO : for 박람회,       point평균값 적용    # 2022-12-12
            # if (idx < 2) or (idx > len(self.point_dict["midrid_point_coordinate"])-3):  slope, _ = get_slope_alpha(x_this, y_this, x_next, y_next) 
            # else : 
            #     # 앞 뒤 각 2개의 point에 대한 기울기를 계산 후 평균값 적용
            #     slope_sum = 0
            #     for num, i in enumerate(range(idx -2, idx + 2)):        
            #         x_tmp_this, y_tmp_this = self.point_dict["midrid_point_coordinate"][i][0], self.point_dict["midrid_point_coordinate"][i][1]
            #         x_tmp_next, y_tmp_next = self.point_dict["midrid_point_coordinate"][i+1][0], self.point_dict["midrid_point_coordinate"][i+1][1]
            #         tmp_slope, _ = get_slope_alpha(x_tmp_this, y_tmp_this, x_tmp_next, y_tmp_next)
            #         slope_sum += tmp_slope
            #     slope = slope_sum/(num+1)
            # -----
            
            # slope, _ = get_slope_alpha(x_this, y_this, x_next, y_next)   # 롤백
            
            if slope == 0: continue         # 기울기가 0인경우는 잘못 계산된 것
            
            # -----TODO : for 박람회        # 2022-12-12
            # if self.plant_type in ["melon", "cucumber", "strawberry"] :                  
            #     if (abs(slope) > 20): slope = 100            
            #     elif (abs(slope) < 1/20) : slope = 1/100       
            #     elif (midrid_slope * slope < 0) : continue 
                
            # else:                                       
            #     if (abs(midrid_slope) > 20) or (abs(slope) > 20): slope = 100            
            #     elif (abs(midrid_slope) < 1/20) or (abs(slope) < 1/20) : slope = 1/100       
            #     elif (midrid_slope * slope < 0) : continue   
            # -----                     
                                                                
            # if (abs(midrid_slope) > 10) or (abs(slope) > 20): pass # slope = 100             # midrid의 기울기 (또는 point의 기울기)가 너무 높으면 잎이 수직으로 찍힌 경우. width를 수평으로 긋는다. 
            # elif (abs(midrid_slope) < 1/10) or (abs(slope) < 1/20) : pass # slope = 1/100        # midrid의 기울기 (또는 point의 기울기)가 너무 낮으면 잎이 수평으로 찍힌 경우. width를 수직으로 긋는다.
            # elif (midrid_slope * slope < 0) : continue       # 두 기울기는 서로 음수, 양수 관계여야 한다. 단, midrid_slope가 너무 높거나 낮은 경우는 제외
        
            inverse_slope = (-1)/slope
            
            x_mid, y_mid = (x_this+x_next)//2, (y_this+y_next)//2       # midrid위의 각 point 사이의 중점     
            coordinate_two_point = [[x_this, y_this], [x_mid, y_mid]]   # midrid위의 현재 point와, 중점 point

            for point in coordinate_two_point:                                                  
                x_coordinate, y_coordinate = point 
                alpha = y_coordinate - x_coordinate*(inverse_slope)   # 위에서 계산한 중점에서의 1차 함수의 절편   

                # y = width_slope*x + alpha : midrid위의 각 point로 표현되는 1차함수에 대해 기울기가 90도 차이나는 1차함수
                check_boolean = False 
                for i in range(self.margin_error):
                    continue_check, continue_num, spot_list, skip_num = create_check_flag(leaf_coordinates)
                    for x, y in leaf_coordinates:

                        if self.plant_type in ["paprika", "chilipepper_seed", "cucumber_seed"]:
                            # 2022-12-12
                            if abs(inverse_slope) < 1/20:     
                                if y_coordinate == y : spot_list.append([x, y])
                            elif abs(inverse_slope) > 20:
                                if x_coordinate == x : spot_list.append([x, y])
                                
                            else: spot_list, continue_num, continue_check = self.find_coordinate_slicing(continue_check, x, y, inverse_slope, alpha, spot_list, continue_num, skip_num, 
                                                                                                        False, i)
                            
                        elif self.plant_type in ["melon", "cucumber", "strawberry"] :
                            spot_list, continue_num, continue_check = self.find_coordinate_slicing(continue_check, x, y, inverse_slope, alpha, spot_list, continue_num, skip_num, 
                                                                                                        False, i)
                        # -----
                        # 
                        #                                                                                 
                             
                        if len(spot_list) == 2:
                            length = math.sqrt(math.pow(spot_list[0][0] - spot_list[1][0], 2) + math.pow(spot_list[0][1] - spot_list[1][1], 2))
                            if length < 20 : 
                                spot_list.pop(-1)
                                continue
                            
                            # spot_list[0] : width의 첫 번째 edge point
                            # spot_list[1] : width의 두 번째 edge point
                            # point : width의 두 edge point 사이의 midrid point
                            edge_point_list.append([spot_list[0], spot_list[1], point])
                            check_boolean = True
                            break
                    if not check_boolean: continue
                    else : break
          
        # calculate max length 
        max_length = 0
        max_length_idx = None
        for point_idx, edge_point in enumerate(edge_point_list):
            pt1_e, pt2_e , center_point = edge_point
            # use this code if you draw all each line between two edge point 
            # cv2.line(img, pt1_e, pt2_e, color=(255, 255, 0), thickness = 1)
            # cv2.circle(img, pt1_e, radius=2, color=(0, 255, 255), thickness=-1)
            # cv2.circle(img, pt2_e, radius=2, color=(0, 255, 255), thickness=-1)
            
            
            length_1 = math.sqrt(math.pow(pt1_e[0] - center_point[0], 2) + math.pow(pt1_e[1] - center_point[1], 2))     # midrid point와 width의 첫 번째 edge point 사이의 거리
            length_2 = math.sqrt(math.pow(center_point[0] - pt2_e[0], 2) + math.pow(center_point[1] - pt2_e[1], 2))     # midrid point와 width의 두 번째 edge point 사이의 거리

            if (length_1 > (length_2) * 10) or  ((length_1)* 10 < length_2) :  continue     # # 두 거리의 차이가 심하면 continue
               
            length = math.sqrt(math.pow(pt1_e[0] - pt2_e[0], 2) + math.pow(pt1_e[1] - pt2_e[1], 2))     # length가 가장 높은 것을 선택
            if max_length <= length: 
                max_length = length
                max_length_idx = point_idx

        cross_point = None
        if max_length_idx is not None:
            pt1_fe, pt2_fe, cross_point = edge_point_list[max_length_idx]
            self.point_dict["leaf_width_edge_coordinates"].append(pt1_fe)
            self.point_dict["leaf_width_edge_coordinates"].append(pt2_fe)
            
        else : 
            return False, cross_point

        # slope, _ = get_slope_alpha(pt1_fe[0], pt1_fe[1], pt2_fe[0], pt2_fe[1])
        # print(f"slope ; {slope}")
        
        
        return True, cross_point


        

