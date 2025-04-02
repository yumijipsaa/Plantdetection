from utils import (get_length, get_slope_alpha, create_check_flag, get_width_point)

# 해당 code에서 허용하는 작물 종류
VALID_PLANTS = ["strawberry", "paprika", "melon", "cucumber", "onion", "seeding_tomato", "chilipepper_seed", "chili",
                "cucumber_seed", "tomato", "bokchoy_seed", "lettuce_seed", "watermelon_seed", "tomato_seed"]  # "tomato"

# 각 작물별 object

plant_object_idx_paprika = ["midrib", "leaf", "stem", "flower", "fruit", "cap", "cap_2"]
plant_object_idx_strawberry = ["midrib", "leaf", "stem", "flower", "fruit", "cap", "y_fruit"]
plant_object_idx_melon = ["midrid", "leaf", "stem", "flower", "fruit", "cap", "petiole"]
plant_object_idx_tomato = ["midrib", "leaf", "stem", "flower", "fruit", "cap", "bud_flower", "growing", "leaf_width"]
plant_object_idx_cucumber = ["midrib", "leaf", "stem", "flower", "fruit", "cap", "fruit_top", "fruit_bottom", "growing"]

plant_object_idx_onion = ["midrid", "leaf", "stem", "flower", "fruit", "cap", "stem_leaf", "first_leaf", "first_leaf_list", "stem_leaf_list"]
plant_object_idx_chilipepper_seed = ["midrib", "leaf", "stem", "flower", "fruit", "cap", "bud_flower", "length"]
plant_object_idx_chili = ["midrib", "leaf", "stem", "flower", "fruit", "cap", "length", "bud_flower"]
plant_object_idx_cucumber_seed = ["midrib", "leaf", "rootstock_length", "scion_length", "length"]
plant_object_idx_bokchoy_seed = ["midrib", "leaf"]
plant_object_idx_lettuce_seed = ["midrib", "leaf"]
plant_object_idx_watermelon_seed = ["leaf", "side_leaf", "hide_leaf", "midrib", "flower", "rootstock_length", "scion_length"]
plant_object_idx_tomato_seed = ["flower", "length", "midrib", "leaf", "bud_flower", "side_leaf", "length"]

# 개수를 count하는 object

count_object_strawberry = ['flower', 'y_fruit']
count_object_paprika = ['flower']
count_object_melon = ['flower']
count_object_cucumber = ['flower', 'growing']
count_object_tomato = ['flower', 'growing', 'bud_flower', "growing"]
count_object_onion = ['flower']
count_object_chilipepper_seed = ['flower', "bud_flower"]
count_object_chili = ['flower', "bud_flower"]
count_object_cucumber_seed = []
count_object_bokchoy_seed = []
count_object_lettuce_seed = []
count_object_watermelon_seed = ['flower']
count_object_tomato_seed = ['flower', 'bud_flower']

# 서로 대응관계가 있는 object인지 확인해야 하는 object (외부, 내부 object는 서로 대응관계가 있다.)
# 외부 object       # 예시: (과일fruit== 외부),(꼭지cap== 내부)
outer_objects_paprika = ["leaf", "fruit"]
outer_objects_strawberry = ["leaf", "fruit"]
outer_objects_melon = ["leaf", "fruit"]
outer_objects_cucumber = ["leaf", "fruit"]
outer_objects_tomato = ['fruit', 'leaf']
outer_objects_onion = ["fruit", "stem_leaf"]
outer_objects_chilipepper_seed = ['leaf']
outer_objects_chili = ['leaf']
outer_objects_cucumber_seed = ['leaf']
outer_objects_bokchoy_seed = ['leaf']
outer_objects_lettuce_seed = ['leaf']
outer_objects_watermelon_seed = ['leaf', 'hide_leaf', 'side_leaf']
outer_objects_tomato_seed = ['leaf', 'side_leaf']

inner_objects_paprika = ['midrib', 'cap', 'cap_2']
inner_objects_strawberry = ['midrib', 'cap']
inner_objects_melon = ['midrid', 'cap']
inner_objects_cucumber = ['midrib', 'cap']
inner_objects_tomato = ['midrib', 'cap', 'leaf_width']
inner_objects_onion = ['first_leaf', 'cap']
inner_objects_chilipepper_seed = ['midrib']
inner_objects_chili = ['midrib']
inner_objects_cucumber_seed = ['midrib']
inner_objects_bokchoy_seed = ['midrib']
inner_objects_lettuce_seed = ['midrib']
inner_objects_watermelon_seed = ['midrib']
inner_objects_tomato_seed = ['midrib']

# 최종적으로 계산된 좌표(장, 폭, center point 등)를 얻고자 하는 object
result_objects_paprika = ['leaf', 'fruit', 'stem', 'flower']
result_objects_strawberry = ['leaf', 'fruit', 'stem', 'flower', 'y_fruit']
result_objects_melon = ['leaf', 'fruit', 'stem', 'flower', 'petiole']
result_objects_cucumber = ['leaf', 'fruit', 'stem', 'flower']
result_objects_tomato = ['midrib', 'stem', 'flower', 'leaf_width', 'fruit']
result_objects_onion = ["fruit", "leaf"]
result_objects_chilipepper_seed = ['leaf', "stem"]
result_objects_cucumber_seed = ['leaf', "length", "rootstock_length", "scion_length"]
result_objects_chili = ['leaf', "stem"]
result_objects_bokchoy_seed = ['leaf']
result_objects_lettuce_seed = ['leaf']
result_objects_watermelon_seed = ["leaf","rootstock_length", "scion_length"]
result_objects_tomato_seed = ["leaf", "midrib", "length"]

# 좌표계산과는 관련 없이 가장자리 좌표(segmentation)를 저장할 object
SEG_OBJECT_LIST = ["leaf", "fruit", "stem", "flower", "y_fruit", "petiole", "growing", "bud_flower"]

import math


def get_tomato_center_points(points,
                             object,
                             width_or_height,
                             mid_midrid_center_points=None,
                             img=None):
    if object == "midrid":
        dr = [[2, 2], [3, 1], [100, 3]]
        selected_point_x_sort = select_points(points, "width", dr)
        selected_point_y_sort = select_points(points, "height", dr)

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
                length = get_length(before_point, point_j)
                if min_length > length:
                    min_length = length
                    next_point = point_j
            if next_point is not None:
                point_list_phase_2.append(next_point)
                before_point = next_point

        return point_list_phase_2


    elif object == "first":
        sorted_center_coordinates = get_center_coordinates(points, width_or_height)

        if width_or_height == "height":
            sorted_center_coordinates.sort(key=lambda x: x[1])
        else:
            sorted_center_coordinates.sort()
        result_point = select_valiable_point(sorted_center_coordinates, 100, 10)  # 9/10 만큼의 point를 제거
        return result_point

    elif object == "last":
        dr = [[1.5, 2], [3, 1], [100, 3]]
        selected_point_x_sort = select_points(points, "width", dr)
        selected_point_y_sort = select_points(points, "height", dr)
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
                if min_length > length:
                    min_length = length
                    next_point = point_j

            if next_point is not None:
                point_list_phase_2.append(next_point)
                before_point = next_point

        result_point = point_list_phase_2
        return result_point


# main function: calculate_coordinates
def select_valiable_point(center_coordinates, threshold_length, using_point_ratio):
    # 거리가 threshold_length이상인 point는 전부 지우고,
    # 전체 point중 (using_point_ratio-1)/using_point_ratio비율 만큼 버린다.

    point_list = []
    for i, point_i in enumerate(center_coordinates):
        if i == len(center_coordinates) - 1 or i == 0:
            point_list.append(point_i)

        if using_point_ratio != 1:  # using_point_ratio = 1 이면 버리는 point는 없다.
            if (i + 1) % using_point_ratio != 0:
                continue  # using_point_ratio = 3 이면 2/3 개수만큼은 버린다.

        minimum_point = None
        min_length = 1000000

        for point_j in center_coordinates[i + 1:]:
            length = math.sqrt(math.pow(point_i[0] - point_j[0], 2) + math.pow(point_i[1] - point_j[1], 2))
            if min_length > length:
                min_length = length
                minimum_point = point_j

        if minimum_point is not None:
            if min_length < threshold_length:  # threshold_length = 5 이면 5보다 크거나 같은 거리의 point는 버린다.
                point_list.append(minimum_point)

    return point_list


def x_or_y_center_coordinates(x_coordinates, y_coordinates, width_or_height):
    """
        coordinates : x좌표 list -> other_coordinates : y좌표 list
        coordinates : y좌표 list -> other_coordinates : x좌표 list
    """

    coordinates_no_dupl = list(set(y_coordinates)) if width_or_height == "height" else list(set(x_coordinates))

    temp_list_1 = []
    temp_list_2 = []
    if width_or_height == "height":
        for no_dulp in coordinates_no_dupl:  # slicing
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

        temp_list_2.sort(key=lambda x: x[1])  # sort according to y coordinates
        # temp_list_2[N][1] : y coordinates of segmentation boundary (removed redundant elements)
        # temp_list_2[N][0] : x coordinates of segmentation boundary (removed redundant elements)
        # 각각의 y coordiante 에 대응되는 x coodinate는 1개 또는 2개 이상이다.

    else:
        for no_dulp in coordinates_no_dupl:  # slicing
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


def get_center_coordinates(leaf_coordinates, width_or_height):
    if width_or_height == "width":
        leaf_coordinates.sort()
    else:
        # 각 leaf_coordinates_half의 bbox를 계산 후 한번 더 sort
        leaf_coordinates.sort(key=lambda x: x[1])

    midrid_center_coordinates = []

    x_coordinates, y_coordinates = [], []
    for point in leaf_coordinates:
        x_coordinates.append(point[0])
        y_coordinates.append(point[1])

    temp_list_2 = x_or_y_center_coordinates(x_coordinates, y_coordinates, width_or_height)

    list_for_check_redundant = []
    total_other_value = None
    count = None

    for idx_, center_coor_no_dupl in enumerate(temp_list_2):
        if width_or_height == "height":
            if center_coor_no_dupl[1] not in list_for_check_redundant:
                # 이미 check한 y좌표가 아니라면
                list_for_check_redundant.append(center_coor_no_dupl[1])

                if idx_ == 0:
                    total_other_value = center_coor_no_dupl[0]
                    count = 1
                    continue

                if count == 1:  # y coordiante 에 대응되는 x coodinate가 1개인 경우
                    midrid_center_coordinates.append(center_coor_no_dupl)

                else:  # y coordiante 에 대응되는 x coodinate가 2개 이상인 경우
                    midrid_center_coordinates.append([int(total_other_value / count), center_coor_no_dupl[1]])
                    # int(total_x_value / count) == mean of x coordinate
                    # center_coor_no_dupl[1]  == y coordinate

                total_other_value = center_coor_no_dupl[0]
                count = 1
            else:
                # 이미 check한 y좌표라면, x좌표 값을 add
                total_other_value += center_coor_no_dupl[0]
                count += 1
        else:
            if center_coor_no_dupl[0] not in list_for_check_redundant:
                # 이미 check한 y좌표가 아니라면
                list_for_check_redundant.append(center_coor_no_dupl[0])

                if idx_ == 0:
                    total_other_value = center_coor_no_dupl[1]
                    count = 1
                    continue

                if count == 1:  # y coordiante 에 대응되는 x coodinate가 1개인 경우
                    midrid_center_coordinates.append(center_coor_no_dupl)

                else:  # y coordiante 에 대응되는 x coodinate가 2개 이상인 경우
                    midrid_center_coordinates.append([center_coor_no_dupl[0], int(total_other_value / count)])
                    # int(total_x_value / count) == mean of x coordinate
                    # center_coor_no_dupl[1]  == y coordinate

                total_other_value = center_coor_no_dupl[1]
                count = 1
            else:
                # 이미 check한 y좌표라면, x좌표 값을 add
                total_other_value += center_coor_no_dupl[1]
                count += 1

    return midrid_center_coordinates


def select_points(points, width_or_height, dr):
    sorted_center_points = get_center_coordinates(points, width_or_height)
    sorted_selected_points = select_valiable_point(sorted_center_points, dr[0][0], dr[0][1])
    sorted_selected_points = select_valiable_point(sorted_selected_points, dr[1][0], dr[1][1])
    sorted_selected_points = select_valiable_point(sorted_selected_points, dr[2][0], dr[2][1])
    return sorted_selected_points


def compute_width_height(bbox):
    x_min, y_min, x_max, y_max = bbox
    return x_max - x_min, y_max - y_min


def return_width_or_height(bbox):
    width, height = compute_width_height(bbox)
    more_longer = "width" if width > height else "height"
    return more_longer


def compute_center_point(bbox):
    x_min, y_min, x_max, y_max = bbox
    return [int((x_min + x_max) / 2), int((y_min + y_max) / 2)]


def get_sorted_center_points(points, width_or_height=None):
    """
    calculate midird coordinates and return a list of midird coordinates
    """

    x_coordinates, y_coordinates = [], []
    for point in points:
        x_coordinates.append(point[0])
        y_coordinates.append(point[1])

    sorted_coordinates = x_or_y_center_coordinates(x_coordinates, y_coordinates, width_or_height)

    midrid_center_coordinates = []
    list_for_check_redundant = []
    total_other_value = None
    count = None

    for idx_, point in enumerate(sorted_coordinates):
        if width_or_height == "height":
            if point[1] not in list_for_check_redundant:
                # 이미 check한 y좌표가 아니라면
                list_for_check_redundant.append(point[1])

                if idx_ == 0:
                    total_other_value = point[0]
                    count = 1
                    continue

                if count == 1:  # y coordiante 에 대응되는 x coodinate가 1개인 경우
                    midrid_center_coordinates.append(point)

                else:  # y coordiante 에 대응되는 x coodinate가 2개 이상인 경우
                    midrid_center_coordinates.append([int(total_other_value / count), point[1]])
                    # int(total_x_value / count) == mean of x coordinate
                    # center_coor_no_dupl[1]  == y coordinate

                total_other_value = point[0]
                count = 1
            else:
                # 이미 check한 y좌표라면, x좌표 값을 add
                total_other_value += point[0]
                count += 1
        else:
            if point[0] not in list_for_check_redundant:
                # 이미 check한 y좌표가 아니라면
                list_for_check_redundant.append(point[0])

                if idx_ == 0:
                    total_other_value = point[1]
                    count = 1
                    continue

                if count == 1:  # y coordiante 에 대응되는 x coodinate가 1개인 경우
                    midrid_center_coordinates.append(point)

                else:  # y coordiante 에 대응되는 x coodinate가 2개 이상인 경우
                    midrid_center_coordinates.append([point[0], int(total_other_value / count)])
                    # int(total_x_value / count) == mean of x coordinate
                    # center_coor_no_dupl[1]  == y coordinate

                total_other_value = point[1]
                count = 1
            else:
                # 이미 check한 y좌표라면, x좌표 값을 add
                total_other_value += point[1]
                count += 1

    if width_or_height == "height":
        midrid_center_coordinates.sort(key=lambda x: x[1])
    else:
        midrid_center_coordinates.sort()

    return midrid_center_coordinates


def get_cucumber_mid_points(points, point_num, get_first):
    # get mid_points (mid_points == length of cucumber)
    mid_points_list_step_1 = []  # cucumber의 높이에 따른 x좌표의 중점을 저장할 list
    num_of_point = point_num  # cucumber의 length를 몇 개의 point를 나눌건지 결정하는 변수
    point_count = int(len(points) / num_of_point)
    count_down = False

    keep_point = None
    exist_point = False
    for i, coordinate in enumerate(points):
        if get_first:
            if i == 0:  # first point
                mid_points_list_step_1.append(tuple(coordinate))
                count_down = True

        if i == len(points) - 1:  # last point
            mid_points_list_step_1.append(tuple(coordinate))

        if count_down:
            point_count -= 1
            if point_count == 0:
                count_down = False
                point_count = int(len(points) / num_of_point)
            else:
                continue

        if keep_point == None:
            keep_point = coordinate
            exist_point = False
        else:
            if exist_point:
                continue
            elif keep_point == coordinate:
                continue
            else:
                x_new, y_new = coordinate
                x_b, y_b = keep_point

                center_fruit_point = (int((x_new + x_b) / 2), int((y_new + y_b) / 2))
                mid_points_list_step_1.append(center_fruit_point)
                exist_point = True
                keep_point = None
                count_down = True

    mid_points_list_step_2 = []
    for idx in range(len(mid_points_list_step_1)):
        if idx == 0 or idx == len(mid_points_list_step_1) - 1:
            mid_points_list_step_2.append(mid_points_list_step_1[idx])
            continue
        x_before, y_before = mid_points_list_step_1[idx - 1][0], mid_points_list_step_1[idx - 1][1]
        x_this, y_this = mid_points_list_step_1[idx][0], mid_points_list_step_1[idx][1]
        x_next, y_next = mid_points_list_step_1[idx + 1][0], mid_points_list_step_1[idx + 1][1]

        slope_before, _ = get_slope_alpha(x_before, y_before, x_this, y_this)
        slope_next, _ = get_slope_alpha(x_this, y_this, x_next, y_next)
        if slope_before * slope_next < 0:
            x_this, y_this = int((x_before + x_next) / 2), int((y_before + y_next) / 2)
            mid_points_list_step_2.append([x_this, y_this])
        else:
            mid_points_list_step_2.append(mid_points_list_step_1[idx])
    # mid_points_list_step_2 = mid_points_list_step_1

    mid_points_list_step_3 = []
    for idx in range(len(mid_points_list_step_2)):
        if idx == 0 or idx == len(mid_points_list_step_2) - 1:
            mid_points_list_step_3.append(mid_points_list_step_2[idx])
            continue
        x_before, y_before = mid_points_list_step_2[idx - 1][0], mid_points_list_step_2[idx - 1][1]
        x_this, y_this = mid_points_list_step_2[idx][0], mid_points_list_step_2[idx][1]
        x_next, y_next = mid_points_list_step_2[idx + 1][0], mid_points_list_step_2[idx + 1][1]
        if (slope_before > 1 and 0 < slope_next < 1) or (slope_next > 1 and 0 < slope_before < 1) or \
                (slope_before < -1 and -1 < slope_next < 0) or (slope_next < -1 and -1 < slope_before < 0):

            x_this, y_this = int((x_before + x_next) / 2), int((y_before + y_next) / 2)
            mid_points_list_step_3.append([x_this, y_this])
        else:
            mid_points_list_step_3.append(mid_points_list_step_2[idx])

    mid_points_list_step_4 = []
    for idx in range(len(mid_points_list_step_3)):
        if idx == len(mid_points_list_step_3) - 1:
            mid_points_list_step_4.append(mid_points_list_step_3[idx])
            break
        mid_points_list_step_4.append(mid_points_list_step_3[idx])
        x_this, y_this = mid_points_list_step_3[idx][0], mid_points_list_step_3[idx][1]
        x_next, y_next = mid_points_list_step_3[idx + 1][0], mid_points_list_step_3[idx + 1][1]
        mid_points_list_step_4.append([int((x_this + x_next) / 2), int((y_this + y_next) / 2)])

    return mid_points_list_step_4


def get_midpoint_between_two_point_2(point_list, distance, first_point=None, last_point=None, img=None):
    """
    point_list를 탐색하며 두 점 사이의 거리가 distance 보다 작을 시 그 중점을 append한다.
    """
    next_point_list = []

    append_flag = True
    for i, point_i in enumerate(point_list):
        if first_point == True and i == 0:
            next_point_list.append(point_i)
            continue
        if last_point == True and i == len(point_list) - 1:
            next_point_list.append(point_i)
            break

        minimum_point = None
        min_length = 10000

        for point_j in point_list[i + 1:]:
            length = math.sqrt(math.pow(point_i[0] - point_j[0], 2) + math.pow(point_i[1] - point_j[1], 2))
            if min_length > length:
                min_length = length
                minimum_point = point_j

        # if min_length == 0.0: continue

        if minimum_point is not None:
            if min_length < distance:  # 10보다 작은 거리에 있는 점들은 그 중점을 append한다.
                next_point_list.append(
                    [int((point_i[0] + minimum_point[0]) / 2), int((point_i[1] + minimum_point[1]) / 2)])
                append_flag = False
            else:
                if append_flag:
                    next_point_list.append(point_i)
                append_flag = True

    return next_point_list
