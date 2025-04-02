import os, json, sys
import cv2
import numpy as np
import math
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.catalog import _MetadataCatalog

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from scipy.optimize import fsolve


def get_width_point(width_points, num_point):
    # 두 개인 width양 끝 점을 기준으로 직선을 그어, 직선상의 num_point만큼의 point를 얻는다
    width_points_list = []

    width_point_1, width_point_2 = width_points
    space_num = num_point - 1

    for i in range(num_point):
        point_x = width_point_1[0] * ((space_num - i) / space_num) + width_point_2[0] * ((i) / space_num)
        point_y = width_point_1[1] * ((space_num - i) / space_num) + width_point_2[1] * ((i) / space_num)
        width_points_list.append([int(point_x), int(point_y)])

    return width_points_list


def create_check_flag(coordinates):
    continue_check = True  # create_check_flag 선언된 곳 아래 for문에서 y == slope*x + alpha 에서 int를 적용해 근사값이 같아지는 경우가 있다.
    skip_num = int(len(coordinates) * (1 / 4))  # 이를 대비하여, 한 번 int(y) == int(slope*x + alpha) 조건이 만족하면
    continue_num = skip_num  # int(len(coordinates)*(1/4)) 번을 건너 뛰어 탐색하도록 한다.
    edge_list = []
    return continue_check, continue_num, edge_list, skip_num


def get_slope_alpha(x_1, y_1, x_2, y_2):
    """
    두 점 (x_1, y_1), (x_2, y_2)을 지나는 1차함수의 기울기, 절편을 계산
    """
    if y_1 - y_2 == 0:  # 1차함수가 수평인 경우
        slope = 0
    elif x_1 - x_2 == 0:  # 1차함수가 수직인 경우
        slope = 100
    else:
        slope = (y_1 - y_2) / (x_1 - x_2)
    alpha = y_2 - x_2 * (slope)  # (x_2, y_2)를 지나는 직선의 alpha(y절편)값
    return slope, alpha


def get_length(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def get_max_min_coordi(coordinates):
    x_max, y_max, x_min, y_min = -1, -1, 100000, 100000
    for x, y in coordinates:
        if x_max < x: x_max = x
        if y_max < y: y_max = y
        if x_min > x: x_min = x
        if y_min > y: y_min = y

    return x_max, y_max, x_min, y_min


#def get_length(point_1, point_2):
#    return math.sqrt(math.pow(point_1[0] - point_2[0], 2) + math.pow(point_1[1] - point_2[1], 2))


def get_length_list(coordinates, half_flag, length_sum, length_list, two_point_length, point, bbox, idx):
    x_mid, y_mid = point
    x_min, y_min, x_max, y_max = bbox
    if half_flag == "start":
        fruit_coordinates = coordinates[:int(len(coordinates) / 2)]
    elif half_flag == "end":
        fruit_coordinates = coordinates[int(len(coordinates) / 2):]

    for x, y in fruit_coordinates:
        if (x_max - x_min <= y_max - y_min) and (y == y_mid):
            two_point_length.append([x, y])
        elif (x_max - x_min > y_max - y_min) and (x == x_mid):
            two_point_length.append([x, y])

        if len(two_point_length) == 2:
            length_two_point = get_length(two_point_length, two_point_length)
            length_from_center = get_length(two_point_length, [x_mid, y_mid])
            if length_two_point < length_from_center:
                two_point_length.pop(-1)
            else:
                length_list.append([idx, length_two_point])
                length_sum += length_two_point
                break
    return length_list, length_sum, two_point_length


def get_dataset_dicts(img_dir, dataset_file, d, c_cfg):
    json_file = os.path.join(img_dir, dataset_file)

    with open(json_file) as f:
        imgs_anns = json.load(f)

    train_dataset_dicts_list = []
    val_dataset_dicts_list = []
    c = 0

    if (c_cfg.TRAIN_VAL_RATIO == 1) or (c_cfg.TRAIN_EVALUATION_RATIO == 1):
        divid_num = None
    else:
        if c_cfg.TRAIN_VAL_RATIO == None:  # evaluation
            divid_num = int(len(imgs_anns["metadatas"]) * c_cfg.TRAIN_EVALUATION_RATIO)
        elif c_cfg.TRAIN_EVALUATION_RATIO == None:  # train
            divid_num = int(len(imgs_anns["metadatas"]) * c_cfg.TRAIN_VAL_RATIO)

    for idx, metadata in enumerate(imgs_anns["metadatas"]):
        record = {}
        image_info = metadata["image_info"]
        file_name = image_info["file_name"]  # 2_20200825_093215.jpg
        filename = os.path.join(img_dir, file_name)  # training에 사용할 image의 path

        record["file_name"] = filename  # file의 path(name아님)
        record["image_id"] = image_info["image_id"]
        record["height"] = image_info["height"]
        record["width"] = image_info["width"]

        objs = []

        for _, instance_info in enumerate(metadata["instance_info"]):
            x_min, y_min, x_val, y_val = instance_info["bbox"]  # x_max = x_min + x_val (same y)
            obj = {
                "bbox": [x_min, y_min, x_min + x_val, y_min + y_val],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": instance_info["segmentation"],
                "category_id": instance_info["category_id"],
            }
            objs.append(obj)
            c += 1
        record["annotations"] = objs

        if divid_num is not None:
            if idx % divid_num == 0:
                val_dataset_dicts_list.append(record)
            else:
                train_dataset_dicts_list.append(record)
        else:
            train_dataset_dicts_list.append(record)

    if d == "train":
        return train_dataset_dicts_list
    elif d == "val":
        return val_dataset_dicts_list


def get_class_name(mood, dataset_file, c_cfg):
    json_file = os.path.join(os.path.join(c_cfg.TRAIN_DIR, mood), dataset_file)

    with open(json_file) as f:
        imgs_anns = json.load(f)

    class_name = imgs_anns["classes"]

    return class_name


def get_metadata_dataset_for_train(c_cfg, args):
    '''
    trainconfig의 property을 근거로 train dataset으로부터 classes의 name을 return한다.

    dataset_metadata : Metadata(name = str(c_cfg.DATASET_NAME), classes_name = [class_name_1, class_name_2, ...])
    '''

    val_dataset_dicts_list = None
    train_dir = os.path.join(c_cfg.TRAIN_DIR, args.type)
    DatasetCatalog.register(c_cfg.DATASET_NAME + "_" + "train",
                            lambda d="train": get_dataset_dicts(train_dir, c_cfg.TRAIN_DATASET, "train", c_cfg))
    DatasetCatalog.register(c_cfg.DATASET_NAME + "_" + "val",
                            lambda d="train": get_dataset_dicts(train_dir, c_cfg.TRAIN_DATASET, "val", c_cfg))
    MetadataCatalog.get(c_cfg.DATASET_NAME + "_" + "train").set(
        thing_classes=get_class_name(args.type, c_cfg.TRAIN_DATASET, c_cfg))

    class_name = get_class_name(args.type, c_cfg.TRAIN_DATASET, c_cfg)
    class_num = len(class_name)

    train_dataset_dicts_list = get_dataset_dicts(train_dir, c_cfg.TRAIN_DATASET, "train", c_cfg)
    val_dataset_dicts_list = get_dataset_dicts(train_dir, c_cfg.TRAIN_DATASET, "val", c_cfg)
    total_images_num = len(train_dataset_dicts_list)

    # print(MetadataCatalog.get(dataset_name + "_train"))
    dataset_metadata = MetadataCatalog.get(c_cfg.DATASET_NAME + "_train")

    return dataset_metadata, class_num, val_dataset_dicts_list, total_images_num


def set_doc(Metadata):
    Metadata.__doc__ = (
            _MetadataCatalog.__doc__
            + """
        .. automethod:: detectron2.data.catalog.MetadataCatalog.get
    """
    )
    return Metadata


def get_metadata_dataset_for_evaluation(c_cfg, args):
    '''
    trainconfig의 property을 근거로 train dataset으로부터 classes의 name을 return한다.

    dataset_metadata : Metadata(name = str(c_cfg.DATASET_NAME), classes_name = [class_name_1, class_name_2, ...])
    '''
    val_dataset_dicts_list = None
    train_dir = os.path.join(c_cfg.TRAIN_DIR, args.type)
    DatasetCatalog.register(c_cfg.DATASET_NAME + "_" + "train",
                            lambda d="train": get_dataset_dicts(train_dir, c_cfg.TRAIN_DATASET, "train", c_cfg))
    DatasetCatalog.register(c_cfg.DATASET_NAME + "_" + "val",
                            lambda d="train": get_dataset_dicts(train_dir, c_cfg.TRAIN_DATASET, "val", c_cfg))

    class_name_gt = get_class_name(args.type, c_cfg.TRAIN_DATASET, c_cfg)
    class_num = len(class_name_gt)

    val_dataset_dicts_list = get_dataset_dicts(train_dir, c_cfg.TRAIN_DATASET, "val", c_cfg)

    metadata_gt = _MetadataCatalog()
    metadata_gt = set_doc(metadata_gt).get(c_cfg.DATASET_NAME + "_" + "train").set(thing_classes=class_name_gt)

    if args.json is not None:
        json_name = args.json
    else:
        json_name = ""
        for word in args.model.split("_"):
            if word == args.type.split("_")[0]:  break
            json_name = json_name + word + "_"
        json_name = json_name + args.type + ".json"

    json_file = os.path.join(os.path.join(os.getcwd(), c_cfg.JSON_PATH_FOR_EVALUATION), json_name)
    with open(json_file) as f:
        class_names_infer = json.load(f)

    metadata_infer = _MetadataCatalog()
    metadata_infer = set_doc(metadata_infer).get(c_cfg.DATASET_NAME + "_" + "train").set(
        thing_classes=class_names_infer)

    return metadata_infer, metadata_gt, class_num, val_dataset_dicts_list, class_name_gt


def get_metadata_for_inference(c_cfg, args):
    # inference후 visualization을 위해 필요한 metadata를 담은 file

    # model name example : 2022-03-25_1617_model_paprika_500000.pth
    # for inference json file name example : 2022-03-25_1617_paprika.json
    for_inference_dir_path = os.path.join(os.getcwd(), c_cfg.JSON_PATH_FOR_INFERENCE)
    yy_mm_dd_hhmm = args.model.split("_")[0] + "_" + args.model.split("_")[1]  # 2022-03-25_1617
    json_file_name = yy_mm_dd_hhmm + "_" + args.type + ".json"  # 2022-03-25_1617_paprika.json
    json_file = os.path.join(for_inference_dir_path, json_file_name)

    if json_file_name not in os.listdir(for_inference_dir_path):
        print("# for inference file dose not exist!")
        print(f"# path : {json_file}")

    with open(json_file) as f:
        metadata = json.load(f)

    dataset_metadata = MetadataCatalog.get(c_cfg.DATASET_NAME + "_" + "train")
    dataset_metadata.set(thing_classes=metadata)

    classes = dataset_metadata.get("thing_classes", None)
    num_classes = len(classes)

    return dataset_metadata, classes, num_classes


def draw_coordinates(coordinates_dict, img):
    # coordinates_dict : {leaf, fruit, stem, flower}

    radius = 4

    if coordinates_dict.get("onion_leaf_num", None) is not None:
        # len(coordinates_dict["onion_leaf_num"]) : number of detected onion pseudostems
        # coordinates_dict["onion_leaf_num"][N] : {"main_center_point", "count_leaf", "leaf_center_points"}
        #   main_center_point = [x_center, y_center]
        #   count_leaf = N (int)
        #   leaf_center_points = [[x_center_1, y_center_1], [x_center_2, y_center_2], ..., [x_center_N, y_center_N]]
        for point_dict in coordinates_dict["onion_leaf_num"]:
            cv2.circle(img, point_dict['main_center_point'], thickness=-1, radius=4, color=(125, 255, 0))

            for i, leaf_center_points in enumerate(point_dict['leaf_center_points']):
                cv2.circle(img, leaf_center_points, thickness=-1, radius=2, color=(255, 125, 0))
                cv2.putText(img, text=f'{i + 1}',
                            org=leaf_center_points,
                            fontFace=cv2.FONT_HERSHEY_DUPLEX,
                            fontScale=0.5,
                            color=(255, 125, 0), thickness=1,
                            lineType=cv2.LINE_AA,
                            bottomLeftOrigin=True)

    if coordinates_dict.get("leaf", None) is not None:
        # len(coordinates_dict["leaf"]) : 측정 가능한 leaf의 개수
        # coordinates_dict["leaf"][N] : {'midrid_point_coordinate', 'leaf_width_edge_coordinates', 'segmentation'}
        # midrid_point_coordinate : [[x_1, y_1], [x_2, y_2], ...]
        # leaf_width_edge_coordinates : [[x_right, y_right], [x_left, y_left]]

        for point_dict in coordinates_dict["leaf"]:
            if "midrid_point_coordinate" in list(point_dict.keys()):
                if point_dict['type'] == 'onion':
                    img = draw_points_line(point_dict["midrid_point_coordinate"], 2, (255, 0, 0), img)
                else:
                    img = draw_points_line(point_dict["midrid_point_coordinate"], radius, (255, 0, 0), img)

            if "leaf_width_edge_coordinates" in list(point_dict.keys()):
                if len(point_dict["leaf_width_edge_coordinates"]) == 0: continue

                img = draw_points_line(point_dict["leaf_width_edge_coordinates"], radius, (0, 0, 255), img)

            if "center" in list(point_dict.keys()):
                cv2.circle(img, point_dict["center"], thickness=-1, radius=4, color=(255, 255, 0))

    if coordinates_dict.get("fruit", None) is not None:
        ### coordinates_dict["fruit"] : {"cap_fruit_side", "cap_fruit_above", "fruit_only"}
        ## cap_fruit_side : [ {"height", "width", "segmentation"}, {"height", "width", "segmentation"}, ...]
        # cap_fruit_side[N]["height"] : [[x_1, y_1], [x_2, y_2]]
        # cap_fruit_side[N]["width"] : [[x_1, y_1], [x_2, y_2]]
        ## cap_fruit_above : : [ {"width", "segmentation"}, {"width", "segmentation"}, ...]
        # cap_fruit_above[N]["width"] : [[x_1, y_1], [x_2, y_2]]
        ## fruit_only : [ {"height", "width", "segmentation"}, {"height", "width", "segmentation"}, ...]
        # fruit_only[N]["height"] : [[x_1, y_1], [x_2, y_2]]
        # fruit_only[N]["width"] : [[x_1, y_1], [x_2, y_2]]
        if 'cap_fruit_side' in list(coordinates_dict["fruit"]):
            for tmp_dict in coordinates_dict["fruit"]["cap_fruit_side"]:
                img = draw_points_line(tmp_dict["width"], radius, (0, 0, 255), img)
                img = draw_points_line(tmp_dict["height"], radius, (255, 0, 0), img)

                cv2.circle(img, tmp_dict["center"], thickness=-1, radius=4, color=(255, 255, 0))

        if coordinates_dict["fruit"].get("fruit_only", None) is not None:
            for tmp_dict in coordinates_dict["fruit"]["fruit_only"]:
                img = draw_points_line(tmp_dict["width"], radius, (0, 0, 255), img)
                img = draw_points_line(tmp_dict["height"], radius, (255, 0, 0), img)

                cv2.circle(img, tmp_dict["center"], thickness=-1, radius=4, color=(255, 255, 0))

                if "vertical_point" in tmp_dict.keys():
                    # cucumber의 fruit 수직 거리
                    # fruit_point_dict["fruit_only"][N]["vertical_point"] = {"first_point", "last_point"}
                    cv2.circle(img, (tmp_dict["vertical_point"]["first_point"]), thickness=-1, color=(255, 0, 0),
                               radius=radius)
                    cv2.circle(img, (tmp_dict["vertical_point"]["last_point"]), thickness=-1, color=(255, 0, 0),
                               radius=radius)
                    cv2.line(img, tmp_dict["vertical_point"]["first_point"], tmp_dict["vertical_point"]["last_point"],
                             thickness=1, color=())
                    cv2.line(img, tmp_dict["vertical_point"]["first_point"], tmp_dict['height'][0], thickness=1,
                             color=())
                    cv2.line(img, tmp_dict["vertical_point"]["last_point"], tmp_dict['height'][-1], thickness=1,
                             color=())

        if coordinates_dict["fruit"].get("cap_fruit_above", None) is not None:
            for tmp_dict in coordinates_dict["fruit"]["cap_fruit_above"]:
                img = draw_points_line(tmp_dict["width"], radius, (0, 0, 255), img)
                img = draw_points_line(tmp_dict["height"], radius, (255, 0, 0), img)

                cv2.circle(img, tmp_dict["center"], thickness=-1, radius=4, color=(255, 255, 0))

    if coordinates_dict.get("stem", None) is not None:
        # len(coordinates_dict["stem"]) : stem의 개수
        # coordinates_dict["stem"][N] = {'width', 'height', 'segmentation'}
        # width : [[x_1, y_1], [x_2, y_2]]
        # hiehgt : [[x_1, y_1], [x_2, y_2]]
        for stem_point in coordinates_dict["stem"]:
            if "width" in stem_point.keys():
                img = draw_points_line(stem_point['width'], radius, (0, 0, 255), img)

            if "height" in stem_point.keys():
                img = draw_points_line(stem_point['height'], radius, (255, 0, 0), img)

            if "center" in stem_point.keys():
                cv2.circle(img, stem_point["center"], thickness=-1, radius=4, color=(255, 255, 0))

    if coordinates_dict.get("petiole", None) is not None:
        if len(coordinates_dict["petiole"]) != 0:
            for i, petiole_point in enumerate(coordinates_dict["petiole"]):
                img = draw_points_line(petiole_point["petiole_point_coordinate"], radius, (0, 0, 255), img)
                cv2.circle(img, petiole_point["center"], thickness=-1, radius=4, color=(255, 255, 0))

    if coordinates_dict.get("y_fruit", None) is not None:
        for i, bbox in enumerate(coordinates_dict['y_fruit']["bbox"]):
            x_min, y_min, x_max, y_max = bbox
            cv2.circle(img, [(x_min + x_max) // 2, (y_min + y_max) // 2], thickness=-1, radius=4, color=(255, 255, 0))

    if coordinates_dict.get("flower", None) is not None:
        for i, bbox in enumerate(coordinates_dict['flower']["bbox"]):
            x_min, y_min, x_max, y_max = bbox
            cv2.circle(img, [(x_min + x_max) // 2, (y_min + y_max) // 2], thickness=-1, radius=4, color=(255, 255, 0))

    return img


def get_sorted_y_center_list(coordinates_dict, object_type):
    y_center_max_list = []

    if object_type == "fruit":
        fruit_info_dict_list = []
        if "cap_fruit_side" in coordinates_dict["fruit"]:
            fruit_info_dict_list.extend(coordinates_dict["fruit"]["cap_fruit_side"])
        if "fruit_only" in coordinates_dict["fruit"]:
            fruit_info_dict_list.extend(coordinates_dict["fruit"]["fruit_only"])

        for point_dict in fruit_info_dict_list:
            if "bbox" in point_dict:
                _, y_min, _, y_max = point_dict["bbox"]
                y_center = (y_min + y_max) // 2
                y_center_max_list.append([y_center, y_max])

    elif object_type in ["flower", "y_fruit", "growing", "bud_flower"]:
        if "bbox" in coordinates_dict[object_type] and coordinates_dict[object_type]["count"] != 0:
            for bbox in coordinates_dict[object_type]["bbox"]:
                _, y_min, _, y_max = bbox
                y_center = (y_min + y_max) // 2
                y_center_max_list.append([y_center, y_max])

    else:
        bbox_list = []
        for point_dict in coordinates_dict[object_type]:
            continue_flag = False
            if "bbox" in point_dict:
                x_min, y_min, x_max, y_max = point_dict["bbox"]
                if any((x_min == bbox[0] and y_min == bbox[1] and x_max == bbox[2] and y_max == bbox[3]) for bbox in bbox_list):
                    continue_flag = True
                if continue_flag:
                    continue

                bbox_list.append([x_min, y_min, x_max, y_max])
                y_center = (y_min + y_max) // 2
                y_center_max_list.append([y_center, y_max])

    y_center_max_list.sort(reverse=True)
    return y_center_max_list


def draw_points_line(point_list, radius, color, img):
    before_point = None
    for point in point_list:
        cv2.circle(img, point, radius=radius, color=color, thickness=-1)
        if before_point is None:
            before_point = point
        else:
            cv2.line(img, before_point, point, color=(), thickness=1)
            before_point = point
    return img


def append_object_number(point_dict, tmp_dict, y_center_max_list, object_type=None):
    if object_type is not None and object_type in ["flower", "y_fruit", "growing", "bud_flower"]:
        _, y_min, _, y_max = point_dict
        y_center = (y_min + y_max) // 2
    else:
        if "bbox" in point_dict:
            _, y_min, _, y_max = point_dict["bbox"]
            y_center = (y_min + y_max) // 2
        else:
            return tmp_dict

    for i, (sorted_y_center, sorted_y_max) in enumerate(y_center_max_list, start=1):
        if [y_center, y_max] == [sorted_y_center, sorted_y_max]:
            if (i + 1) not in tmp_dict['number']:  # 중복 확인 후 추가
                tmp_dict['number'].append(i)
                break

    return tmp_dict


def delete_duplicate_values(bbox, bbox_list):
    x_min, x_max, y_min, y_max = bbox
    continue_flag = False
    for box in bbox_list:  # 완전히 같은 bbox는 제외
        if not (x_min - box[0] and y_min - box[1] and x_max - box[2] and y_max - box[3]):
            continue_flag = True

    return continue_flag


def save_to_json(coordinates_dict, segmentations):
    json_file = {}

    # json_file['onion_leaf_num'] : list
    # json_file['onion_leaf_num'][N]['stem_center'] = [x_center, y_center]
    # json_file['onion_leaf_num'][N]['stem_num'] = int
    # json_file['onion_leaf_num'][N]['num_leaf'] = int
    # json_file['onion_leaf_num'][N]['leaf_num'] = int
    # json_file['onion_leaf_num'][N]['leaf_center'] : list
    # json_file['onion_leaf_num'][N]['leaf_center'][N] : [x_center, y_center]
    if coordinates_dict.get("onion_leaf_num", None) is not None:
        json_file['onion_leaf_num'] = []
        for i, point_dict in enumerate(coordinates_dict["onion_leaf_num"]):
            tmp_dict = {}
            tmp_dict['stem_center'] = point_dict['main_center_point']
            tmp_dict['stem_num'] = i + 1
            tmp_dict['num_leaf'] = len(point_dict['leaf_center_points'])

            tmp_dict['leaf_center'] = []
            for j, leaf_center_points in enumerate(point_dict['leaf_center_points']):
                tmp_dict['leaf_center'].append(leaf_center_points)

                tmp_dict['leaf_num'] = j + 1

        json_file['onion_leaf_num'].append(tmp_dict)

    # json_file['leaf'] : list
    # json_file['leaf'][N]['height'] : [{'x': 619, 'y': 904}, {'x': 616, 'y': 909}, ... ,{'x': 542, 'y': 1096}]
    # json_file['leaf'][N]['width'] : [{'x': 508, 'y': 960}, {'x': 654, 'y': 1015}]
    # json_file['leaf'][N]['segmentation'] : [{'x': 748, 'y': 3288}, {'x': 747, 'y': 3289}, ...]
    if coordinates_dict.get("leaf", None) is not None:
        json_file['leaf'] = []
        bbox_list = []
        y_center_max_list = get_sorted_y_center_list(coordinates_dict, 'leaf')
        for i, point_dict in enumerate(coordinates_dict["leaf"]):
            tmp_dict = {}
            tmp_dict['height'] = []
            tmp_dict['width'] = []
            tmp_dict['center'] = []
            tmp_dict['segmentation'] = []
            tmp_dict['type'] = point_dict['type']
            if point_dict['bbox'] is not None:
                continue_flag = delete_duplicate_values(point_dict['bbox'], bbox_list)
                if continue_flag: continue
                bbox_list.append(point_dict['bbox'])

            if "midrid_point_coordinate" in list(point_dict.keys()):
                for _, point in enumerate(point_dict["midrid_point_coordinate"]):
                    tmp_point_midrid = {}
                    tmp_point_midrid['x'], tmp_point_midrid['y'] = point[0], point[1]
                    tmp_dict['height'].append(tmp_point_midrid)

            if "leaf_width_edge_coordinates" in list(point_dict.keys()):
                for width_point in point_dict["leaf_width_edge_coordinates"]:
                    tmp_point_width = {}
                    tmp_point_width['x'], tmp_point_width['y'] = width_point[0], width_point[1]
                    tmp_dict['width'].append(tmp_point_width)

            if "center" in list(point_dict.keys()):
                tmp_center = {}
                tmp_center["x"], tmp_center["y"] = point_dict["center"]
                tmp_dict['center'].append(tmp_center)

            if "segmentation" in list(point_dict.keys()) and point_dict["segmentation"] is not None:
                for seg_x, seg_y in point_dict["segmentation"]:
                    tmp_point_seg = {}
                    tmp_point_seg['x'], tmp_point_seg['y'] = seg_x, seg_y
                    tmp_dict['segmentation'].append(tmp_point_seg)
            tmp_dict["number"] = []
            tmp_dict = append_object_number(point_dict, tmp_dict, y_center_max_list)
            json_file['leaf'].append(tmp_dict)
    if coordinates_dict.get("petiole", None) is not None:
        json_file['petiole'] = []
        bbox_list = []
        y_center_max_list = get_sorted_y_center_list(coordinates_dict, 'petiole')
        for i, point_dict in enumerate(coordinates_dict["petiole"]):
            tmp_dict = {}
            tmp_dict['height'] = []
            tmp_dict['width'] = []
            tmp_dict['segmentation'] = []
            tmp_dict['center'] = []
            tmp_dict['type'] = point_dict['type']

            continue_flag = delete_duplicate_values(point_dict['bbox'], bbox_list)
            if continue_flag: continue
            bbox_list.append(point_dict['bbox'])

            if "petiole_point_coordinate" in list(point_dict.keys()):
                for _, point in enumerate(point_dict["petiole_point_coordinate"]):
                    tmp_point_midrid = {}
                    tmp_point_midrid['x'], tmp_point_midrid['y'] = point[0], point[1]
                    tmp_dict['height'].append(tmp_point_midrid)

            if "center" in list(point_dict.keys()):
                tmp_center = {}
                tmp_center["x"], tmp_center["y"] = point_dict["center"]
                tmp_dict['center'].append(tmp_center)

            if "segmentation" in list(point_dict.keys()):
                for seg_x, seg_y in point_dict["segmentation"]:
                    tmp_point_seg = {}
                    tmp_point_seg['x'], tmp_point_seg['y'] = seg_x, seg_y
                    tmp_dict['segmentation'].append(tmp_point_seg)

            tmp_dict["number"] = []
            tmp_dict = append_object_number(point_dict, tmp_dict, y_center_max_list)

            json_file['petiole'].append(tmp_dict)

    # json_file['fruit'] : list
    # json_file['fruit'][N]['width'] : [{'x': 404, 'y': 1122}, {'x': 531, 'y': 1132}]
    # json_file['fruit'][N]['height'] : [{'x': 467, 'y': 1091}, {'x': 430, 'y': 1214}]
    # json_file['fruit'][N]['segmentation'] : [{'x': 748, 'y': 3288}, {'x': 747, 'y': 3289}, ...]
    if coordinates_dict.get("fruit", None) is not None:
        json_file['fruit'] = []
        bbox_list = []
        y_center_max_list = get_sorted_y_center_list(coordinates_dict, 'fruit')
        if 'cap_fruit_side' in list(coordinates_dict["fruit"]):
            for fruit_dict in coordinates_dict["fruit"]["cap_fruit_side"]:
                tmp_dict = {}
                tmp_dict['width'] = []
                tmp_dict['height'] = []
                tmp_dict['segmentation'] = []
                tmp_dict['center'] = []
                tmp_dict['type'] = fruit_dict['type']

                continue_flag = delete_duplicate_values(fruit_dict['bbox'], bbox_list)
                if continue_flag: continue
                bbox_list.append(fruit_dict['bbox'])

                if "width" in list(fruit_dict.keys()):
                    for width in fruit_dict["width"]:
                        tmp_point_width = {}
                        tmp_point_width['x'], tmp_point_width['y'] = width[0], width[1]
                        tmp_dict['width'].append(tmp_point_width)

                if "height" in list(fruit_dict.keys()):
                    for height in fruit_dict["height"]:
                        tmp_point_height = {}
                        tmp_point_height['x'], tmp_point_height['y'] = height[0], height[1]
                        tmp_dict['height'].append(tmp_point_height)

                if "center" in list(fruit_dict.keys()):
                    tmp_center = {}
                    tmp_center["x"], tmp_center["y"] = fruit_dict["center"]
                    tmp_dict['center'].append(tmp_center)

                if "segmentation" in list(fruit_dict.keys()):
                    for seg_x, seg_y in fruit_dict["segmentation"]:
                        tmp_point_seg = {}
                        tmp_point_seg['x'], tmp_point_seg['y'] = seg_x, seg_y
                        tmp_dict['segmentation'].append(tmp_point_seg)

                if "is_curved" in list(fruit_dict.keys()):
                    tmp_dict['curved'] = fruit_dict["is_curved"]

                if "curved_points" in list(fruit_dict.keys()):
                    tmp_dict['curved_points'] = fruit_dict["curved_points"]

                tmp_dict["number"] = []
                tmp_dict = append_object_number(fruit_dict, tmp_dict, y_center_max_list)

                json_file['fruit'].append(tmp_dict)

        if 'fruit_only' in list(coordinates_dict["fruit"]):
            for fruit_dict in coordinates_dict["fruit"]["fruit_only"]:
                tmp_dict = {}
                tmp_dict['width'] = []
                tmp_dict['height'] = []
                tmp_dict['segmentation'] = []
                tmp_dict['center'] = []
                tmp_dict['type'] = fruit_dict['type']

                continue_flag = delete_duplicate_values(fruit_dict['bbox'], bbox_list)
                if continue_flag: continue
                bbox_list.append(fruit_dict['bbox'])

                if "width" in list(fruit_dict.keys()):
                    for width in fruit_dict["width"]:
                        tmp_point_width = {}
                        tmp_point_width['x'], tmp_point_width['y'] = width[0], width[1]
                        tmp_dict['width'].append(tmp_point_width)

                if "height" in list(fruit_dict.keys()):
                    for height in fruit_dict["height"]:
                        tmp_point_height = {}
                        tmp_point_height['x'], tmp_point_height['y'] = height[0], height[1]
                        tmp_dict['height'].append(tmp_point_height)

                if "center" in list(fruit_dict.keys()):
                    tmp_center = {}
                    tmp_center["x"], tmp_center["y"] = fruit_dict["center"]
                    tmp_dict['center'].append(tmp_center)

                if "segmentation" in list(fruit_dict.keys()):
                    for seg_x, seg_y in fruit_dict["segmentation"]:
                        tmp_point_seg = {}
                        tmp_point_seg['x'], tmp_point_seg['y'] = seg_x, seg_y
                        tmp_dict['segmentation'].append(tmp_point_seg)

                if "is_curved" in list(fruit_dict.keys()):
                    tmp_dict['curved'] = fruit_dict["is_curved"]

                if "curved_points" in list(fruit_dict.keys()):
                    tmp_dict['curved_points'] = fruit_dict["curved_points"]

                tmp_dict["number"] = []
                tmp_dict = append_object_number(fruit_dict, tmp_dict, y_center_max_list)

                json_file['fruit'].append(tmp_dict)

                # json_file['stem'] : list
    # json_file['stem'][N]["width"] : [{'x': 404, 'y': 1122}, {'x': 531, 'y': 1132}]
    # json_file['stem'][N]["height"] : [{'x': 467, 'y': 1091}, {'x': 430, 'y': 1214}]
    # json_file['stem'][N]['segmentation'] : [{'x': 748, 'y': 3288}, {'x': 747, 'y': 3289}, ...]
    if coordinates_dict.get("stem", None) is not None:
        json_file['stem'] = []
        bbox_list = []
        y_center_max_list = get_sorted_y_center_list(coordinates_dict, 'stem')
        for i, stem_dict in enumerate(coordinates_dict["stem"]):
            tmp_dict = {}
            tmp_dict['width'] = []
            tmp_dict['height'] = []
            tmp_dict['segmentation'] = []
            tmp_dict['center'] = []
            tmp_dict['type'] = stem_dict['type']

            ### delete_duplicate_values 버그 수정 함수 적용 (stem에만)
            continue_flag = delete_duplicate_values_stem(stem_dict['bbox'], bbox_list)
            if continue_flag: continue
            bbox_list.append(stem_dict['bbox'])

            if "width" in list(stem_dict.keys()):
                for width in stem_dict['width']:
                    tmp_point_width = {}
                    tmp_point_width['x'], tmp_point_width['y'] = width[0], width[1]
                    tmp_dict['width'].append(tmp_point_width)

            if "height" in list(stem_dict.keys()):
                for height in stem_dict["height"]:
                    tmp_point_height = {}
                    tmp_point_height['x'], tmp_point_height['y'] = height[0], height[1]
                    tmp_dict['height'].append(tmp_point_height)

            if "segmentation" in list(stem_dict.keys()):
                for seg_x, seg_y in stem_dict["segmentation"]:
                    tmp_point_seg = {}
                    tmp_point_seg['x'], tmp_point_seg['y'] = seg_x, seg_y
                    tmp_dict['segmentation'].append(tmp_point_seg)

            if "center" in list(stem_dict.keys()):
                tmp_center = {}
                tmp_center["x"], tmp_center["y"] = stem_dict["center"]
                tmp_dict['center'].append(tmp_center)

            tmp_dict["number"] = []
            tmp_dict = append_object_number(stem_dict, tmp_dict, y_center_max_list)
            
            ### stem이 growing을 포함하고 있는 지 여부를 bool type으로 추가
            if "contain_growing" in list(stem_dict.keys()):
                tmp_dict["contain_growing"] = stem_dict["contain_growing"]  

            json_file['stem'].append(tmp_dict)
    # json_file["flower"] : list
    # json_file["flower"][N] : {'x_min': 639, 'x_max': 581, 'y_min': 671, 'y_max': 613}
    if coordinates_dict.get("flower", None) is not None:
        json_file["flower"] = []

        y_center_max_list = get_sorted_y_center_list(coordinates_dict, 'flower')

        for i, bbox in enumerate(coordinates_dict['flower']["bbox"]):
            tmp_dict = {}
            tmp_dict["segmentation"] = []  # 실제로는 bbox임

            x_min, y_min, x_max, y_max = bbox

            tmp_dict["segmentation"].append(dict({'x': x_min, "y": y_min}))
            tmp_dict["segmentation"].append(dict({'x': x_max, "y": y_min}))
            tmp_dict["segmentation"].append(dict({'x': x_max, "y": y_max}))
            tmp_dict["segmentation"].append(dict({'x': x_min, "y": y_max}))
            tmp_dict["center"] = [dict({'x': (x_min + x_max) // 2, "y": (y_min + y_max) // 2})]

            tmp_dict["width"] = []
            tmp_dict["height"] = []

            tmp_dict["number"] = []
            if y_center_max_list is not None:
                tmp_dict = append_object_number(bbox, tmp_dict, y_center_max_list, "flower")
            json_file["flower"].append(tmp_dict)

    if coordinates_dict.get("growing", None) is not None:
        json_file["growing"] = []

        y_center_max_list = get_sorted_y_center_list(coordinates_dict, 'growing')

        for i, bbox in enumerate(coordinates_dict['growing']["bbox"]):
            tmp_dict = {}
            tmp_dict["segmentation"] = []  # 실제로는 bbox임

            x_min, y_min, x_max, y_max = bbox

            tmp_dict["segmentation"].append(dict({'x': x_min, "y": y_min}))
            tmp_dict["segmentation"].append(dict({'x': x_max, "y": y_min}))
            tmp_dict["segmentation"].append(dict({'x': x_max, "y": y_max}))
            tmp_dict["segmentation"].append(dict({'x': x_min, "y": y_max}))
            tmp_dict["center"] = [dict({'x': (x_min + x_max) // 2, "y": (y_min + y_max) // 2})]

            tmp_dict["width"] = []
            tmp_dict["height"] = []

            tmp_dict["number"] = []
            if y_center_max_list is not None:
                tmp_dict = append_object_number(bbox, tmp_dict, y_center_max_list, "growing")
            json_file["growing"].append(tmp_dict)

    if coordinates_dict.get("bud_flower", None) is not None:
        json_file["bud_flower"] = []

        y_center_max_list = get_sorted_y_center_list(coordinates_dict, 'bud_flower')

        for i, bbox in enumerate(coordinates_dict['bud_flower']["bbox"]):
            tmp_dict = {}
            tmp_dict["segmentation"] = []  # 실제로는 bbox임

            x_min, y_min, x_max, y_max = bbox

            tmp_dict["segmentation"].append(dict({'x': x_min, "y": y_min}))
            tmp_dict["segmentation"].append(dict({'x': x_max, "y": y_min}))
            tmp_dict["segmentation"].append(dict({'x': x_max, "y": y_max}))
            tmp_dict["segmentation"].append(dict({'x': x_min, "y": y_max}))
            tmp_dict["center"] = [dict({'x': (x_min + x_max) // 2, "y": (y_min + y_max) // 2})]

            tmp_dict["width"] = []
            tmp_dict["height"] = []

            tmp_dict["number"] = []
            if y_center_max_list is not None:
                tmp_dict = append_object_number(bbox, tmp_dict, y_center_max_list, "bud_flower")
            json_file["bud_flower"].append(tmp_dict)

    if coordinates_dict.get("y_fruit", None) is not None:
        json_file["y_fruit"] = []
        y_center_max_list = get_sorted_y_center_list(coordinates_dict, 'y_fruit')
        for i, bbox in enumerate(coordinates_dict['y_fruit']["bbox"]):
            tmp_dict = {}
            tmp_dict["segmentation"] = []  # 실제로는 bbox임

            x_min, y_min, x_max, y_max = bbox

            tmp_dict["segmentation"].append(dict({'x': x_min, "y": y_min}))
            tmp_dict["segmentation"].append(dict({'x': x_max, "y": y_min}))
            tmp_dict["segmentation"].append(dict({'x': x_max, "y": y_max}))
            tmp_dict["segmentation"].append(dict({'x': x_min, "y": y_max}))
            tmp_dict["center"] = [dict({'x': (x_min + x_max) // 2, "y": (y_min + y_max) // 2})]

            tmp_dict["width"] = []
            tmp_dict["height"] = []

            tmp_dict["number"] = []
            if y_center_max_list is not None:
                tmp_dict = append_object_number(bbox, tmp_dict, y_center_max_list, "flower")
            json_file["y_fruit"].append(tmp_dict)

    if coordinates_dict.get("rootstock_length", None) is not None:
        json_file['rootstock_length'] = []
        bbox_list = []
        y_center_max_list = get_sorted_y_center_list(coordinates_dict, 'rootstock_length')

        for i, rootstock_length_dict in enumerate(coordinates_dict["rootstock_length"]):
            tmp_dict = {}
            tmp_dict['width'] = []
            tmp_dict['height'] = []
            tmp_dict['segmentation'] = []
            tmp_dict['center'] = []
            tmp_dict['type'] = rootstock_length_dict['type']

            continue_flag = delete_duplicate_values(rootstock_length_dict['bbox'], bbox_list)
            if continue_flag: continue
            bbox_list.append(rootstock_length_dict['bbox'])

            if "width" in list(rootstock_length_dict.keys()):
                for width in rootstock_length_dict['width']:
                    tmp_point_width = {}
                    tmp_point_width['x'], tmp_point_width['y'] = width[0], width[1]
                    tmp_dict['width'].append(tmp_point_width)

            if "height" in list(rootstock_length_dict.keys()):
                for height in rootstock_length_dict["height"]:
                    tmp_point_height = {}
                    tmp_point_height['x'], tmp_point_height['y'] = height[0], height[1]
                    tmp_dict['height'].append(tmp_point_height)

            if "segmentation" in list(rootstock_length_dict.keys()):
                for seg_x, seg_y in rootstock_length_dict["segmentation"]:
                    tmp_point_seg = {}
                    tmp_point_seg['x'], tmp_point_seg['y'] = seg_x, seg_y
                    tmp_dict['segmentation'].append(tmp_point_seg)

            if "center" in list(rootstock_length_dict.keys()):
                tmp_center = {}
                tmp_center["x"], tmp_center["y"] = rootstock_length_dict["center"]
                tmp_dict['center'].append(tmp_center)

            tmp_dict["number"] = []
            tmp_dict = append_object_number(rootstock_length_dict, tmp_dict, y_center_max_list)
            json_file['rootstock_length'].append(tmp_dict)

    if coordinates_dict.get("scion_length", None) is not None:
        json_file['scion_length'] = []
        bbox_list = []
        y_center_max_list = get_sorted_y_center_list(coordinates_dict, 'scion_length')

        for i, scion_length_dict in enumerate(coordinates_dict["scion_length"]):
            tmp_dict = {}
            tmp_dict['width'] = []
            tmp_dict['height'] = []
            tmp_dict['segmentation'] = []
            tmp_dict['center'] = []
            tmp_dict['type'] = scion_length_dict['type']

            continue_flag = delete_duplicate_values(scion_length_dict['bbox'], bbox_list)
            if continue_flag: continue
            bbox_list.append(scion_length_dict['bbox'])

            if "width" in list(scion_length_dict.keys()):
                for width in scion_length_dict['width']:
                    tmp_point_width = {}
                    tmp_point_width['x'], tmp_point_width['y'] = width[0], width[1]
                    tmp_dict['width'].append(tmp_point_width)

            if "height" in list(scion_length_dict.keys()):
                for height in scion_length_dict["height"]:
                    tmp_point_height = {}
                    tmp_point_height['x'], tmp_point_height['y'] = height[0], height[1]
                    tmp_dict['height'].append(tmp_point_height)

            if "segmentation" in list(scion_length_dict.keys()):
                for seg_x, seg_y in scion_length_dict["segmentation"]:
                    tmp_point_seg = {}
                    tmp_point_seg['x'], tmp_point_seg['y'] = seg_x, seg_y
                    tmp_dict['segmentation'].append(tmp_point_seg)

            if "center" in list(scion_length_dict.keys()):
                tmp_center = {}
                tmp_center["x"], tmp_center["y"] = scion_length_dict["center"]
                tmp_dict['center'].append(tmp_center)

            tmp_dict["number"] = []
            tmp_dict = append_object_number(scion_length_dict, tmp_dict, y_center_max_list)
            json_file['scion_length'].append(tmp_dict)

    if coordinates_dict.get("bud_flower", None) is not None:
        json_file["bud_flower"] = []

        y_center_max_list = get_sorted_y_center_list(coordinates_dict, 'bud_flower')

        for i, bbox in enumerate(coordinates_dict['bud_flower']["bbox"]):
            tmp_dict = {}
            tmp_dict["segmentation"] = []  # 실제로는 bbox임

            x_min, y_min, x_max, y_max = bbox

            tmp_dict["segmentation"].append(dict({'x': x_min, "y": y_min}))
            tmp_dict["segmentation"].append(dict({'x': x_max, "y": y_min}))
            tmp_dict["segmentation"].append(dict({'x': x_max, "y": y_max}))
            tmp_dict["segmentation"].append(dict({'x': x_min, "y": y_max}))
            tmp_dict["center"] = [dict({'x': (x_min + x_max) // 2, "y": (y_min + y_max) // 2})]

            tmp_dict["width"] = []
            tmp_dict["height"] = []

            tmp_dict["number"] = []
            if y_center_max_list is not None:
                tmp_dict = append_object_number(bbox, tmp_dict, y_center_max_list, "bud_flower")
            json_file["bud_flower"].append(tmp_dict)

    json_path = os.path.join(os.getcwd(), "json_dict.json")

    json.dump(json_file, open(json_path, "w"), indent=4)

    # for keys in list(json_file.keys()):
    #     num = len(json_file[f'{keys}'])
    #     print(f"type: {keys}, num of {keys} : {num}")
    #     if keys == 'leaf':
    #         for tmp_dict in json_file[f'{keys}']:
    #             list_keys = list(tmp_dict.keys())
    #             for _list_keys in list_keys:
    #                 value = tmp_dict[f'{_list_keys}']
    #                 print(f"key : {_list_keys}, len(value) : {len(value)}, ", end = " ")

    #                 for i, val in enumerate(value):
    #                     if i < 4:
    #                         print(f"value[{i}] : {val}" ,  end = ' ')
    #                     else : break
    #                 print(" ")

    #             print(" ")
    #     else:

    #         for tmp_dict in json_file[f'{keys}']:
    #             list_keys = list(tmp_dict.keys())
    #             for _list_keys in list_keys:

    #                 value = tmp_dict[f'{_list_keys}']
    #                 print(f"{_list_keys}", end = ' ')
    #             print(" ")

    #     print("----")

    return json_file  # 167


# path=r'D:\Downloads\Detectron2_CV_Plant_Analysis\test_images_paprika\20200428_111523.jpg'
# import cv2
# image = cv2.imread(path)
#
# image.shape
# img = resize_image(image)
# img.shape
#
# 3024/1280
# 4032/1706
def resize_image(image):
    resizeing_ratio = 1280 / image.shape[1]
    resize_height = int(image.shape[0] * resizeing_ratio)
    resize_width = int(image.shape[1] * resizeing_ratio)

    if image.shape[0] > 1280:
        resize_image = cv2.resize(image, (resize_width, resize_height), cv2.INTER_AREA)
    elif image.shape[0] == 1280:
        pass
    else:
        resize_image = cv2.resize(image, (resize_width, resize_height), cv2.INTER_CUBIC)

    return resize_image


def resize(img, min_size, max_size, pad=False):
    import cv2
    '''
    resize image, keep aspect ratio.
    Args:
        img: torch tensor, or opencv image (C,H,W)
        min_size: min size of new image
        max_size: max size of new image
    Returns:
        sizeed image, which has padded to match min,max size, the non-padded image area has dimension ratio as original img
        original image size
    '''

    ori_size = img.shape[:2]

    tall = True if ori_size[0] > ori_size[1] else False

    if tall:
        new_size = (max_size, min_size)
        ratio = new_size[0] / float(ori_size[0])
        new_dimension = (new_size[0], int(ori_size[1] * ratio))
    else:
        new_size = (min_size, max_size)
        ratio = new_size[1] / float(ori_size[1])
        new_dimension = (int(ori_size[0] * ratio), new_size[1])

    new_img = cv2.resize(img, dsize=new_dimension)

    # if pad:
    #     topad = [0, 0, new_size[1] - new_dimension[1], new_size[0] - new_dimension[0]]  ### left, top, right, bottom
    #     new_img = F.pad(new_img,topad,fill=255) ## filling with white space
    return new_img, ori_size, ratio


def comfute_iou(infer_box, gt_box):
    """
    infer_box : x_min, y_min, x_max, y_max
    gt_box : x_min, y_min, x_max, y_max
    """
    box1_area = (infer_box[2] - infer_box[0]) * (infer_box[3] - infer_box[1])
    box2_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])

    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(infer_box[0], gt_box[0])  # x_min 중 큰 것
    y1 = max(infer_box[1], gt_box[1])  # y_min 중 큰 것   # (x1, y1) : 두 left_top points 중 큰 값, intersction의 lest_top
    x2 = min(infer_box[2], gt_box[2])  # x_max 중 작은 것
    y2 = min(infer_box[3], gt_box[3])  # y_max 중 작은 것  # (x2, y2) : 두 right_bottom points 중 작은 값  intersction의 right_bottom

    # compute the width and height of the intersection
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)

    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    return iou


def get_box_degree(seg):
    rect = cv2.minAreaRect(seg)
    box = cv2.boxPoints(rect)
    box = box.astype(np.int32)

    p = box[0]
    box = box[1:]

    angle = 0
    if p[0] == box[:, 0].max() or p[0] == box[:, 0].min():  # 수직 이미지
        if box[:, 0].max() - box[:, 0].min() < box[:, 1].max() - box[:, 1].min():
            pass
        else:
            angle = 90
    elif p[0] < box[:, 0].min() or p[0] > box[:, 0].max():  # 좌우 꼭지점
        p1 = box[np.where(box[:, 1] == box[:, 1].max())][0]
        p2 = box[np.where(box[:, 1] == box[:, 1].min())][0]
        if ((p1[0] - p[0]) ** 2 + (p1[1] - p[1]) ** 2) < ((p2[0] - p[0]) ** 2 + (p2[1] - p[1]) ** 2):
            angle = round(math.degrees(math.atan((p1[1] - p[1]) / (p1[0] - p[0]))))
        else:
            angle = round(math.degrees(math.atan((p2[1] - p[1]) / (p2[0] - p[0]))))

    else:  # 상하 꼭지점 # 없는 듯 하다
        p1 = box[np.where(box[:, 0] == box[:, 0].max())][0]
        p2 = box[np.where(box[:, 0] == box[:, 0].min())][0]
        if ((p1[0] - p[0]) ** 2 + (p1[1] - p[1]) ** 2) < ((p2[0] - p[0]) ** 2 + (p2[1] - p[1]) ** 2):
            angle = round(math.degrees(math.atan((p1[1] - p[1]) / (p1[0] - p[0]))))
        else:
            angle = round(math.degrees(math.atan((p2[1] - p[1]) / (p2[0] - p[0]))))

    return angle, rect[0]


def get_skeleton_from_mask(mask):
    _, biimg = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    dst = cv2.distanceTransform(biimg, cv2.DIST_L2, 5)
    cv2.normalize(dst, dst, 0, 255, cv2.NORM_MINMAX)

    dst = dst.astype(np.uint8)
    dst = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, -2)

    # size = np.size(mask)
    # skel = np.zeros(mask.shape, np.uint8)
    #
    # element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    # done = False
    #
    # while not done:
    #     eroded = cv2.erode(mask, element)
    #     temp = cv2.dilate(eroded, element)
    #     temp = cv2.subtract(mask, temp)
    #     skel = cv2.bitwise_or(skel, temp)
    #     mask = eroded.copy()
    #
    #     zeros = size - cv2.countNonZero(mask)
    #     if zeros == size:
    #         done = True

    dots = np.where(dst == 255)
    # dots = np.where(skel == 255)
    x = dots[1][..., np.newaxis]
    y = dots[0][..., np.newaxis]
    return x, y


def get_curve_model(x, y, degree):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    y_poly = poly.fit_transform(y)

    model_y = LinearRegression()
    model_y.fit(y_poly, x)
    return model_y


def linear_in_fruit(model, contours, degree):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    rads = np.where(contours == 255)
    y = rads[0]
    y_range_all = np.unique(y).reshape(-1, 1)
    y_range_all = poly.fit_transform(y_range_all)
    predicts_all = model.predict(y_range_all)
    predicts_all = np.reshape(predicts_all, (-1,))
    y_range_all = y_range_all[:, 0]
    for i in range(len(predicts_all)):
        if predicts_all[i] > contours.shape[1] - 1:
            predicts_all[i] = contours.shape[1] - 1
    for i in range(len(predicts_all)):
        if predicts_all[i] < 0:
            predicts_all[i] = 0
    pt = []
    for i in range(len(predicts_all)):
        if contours[math.floor(y_range_all[i])][math.floor(predicts_all[i])] == 255:
            pt.append(math.floor(y_range_all[i]))
    pt = np.array(pt)
    y_range = np.floor(np.linspace(min(pt), max(pt), 21))[..., np.newaxis]
    y_range = poly.fit_transform(y_range)

    predicts = model.predict(y_range)
    predicts = np.reshape(predicts, (len(predicts),))
    return predicts, y_range


def curve_points(model, x, y):
    poly = PolynomialFeatures(degree=2, include_bias=False)
    p1, p2 = np.array((math.floor(x[0]), math.floor(y[0][0]))), np.array((math.floor(x[-1]), math.floor(y[-1][0])))
    (mx, my) = (p1 + p2) / 2
    if p1[0] == p2[0]:
        p3 = np.array((math.floor(x[10]), math.floor(y[10][0])))
    else:
        slope = (math.floor(x[-1]) - (math.floor(x[0]))) / (math.floor(y[0][0]) - math.floor(y[-1][0]))
        intercept = my - slope * mx

        def line(y):
            return (y - intercept) / slope

        def equation(y):
            return model.predict(poly.fit_transform([[y]]))[0] - line(y)

        initial_guess = my
        intersection_y = fsolve(lambda y: equation(y[0]), np.array([initial_guess]))
        p3 = (math.floor((model.predict(poly.fit_transform([[intersection_y[0]]]))[0][0])), math.floor(intersection_y[0]))

    return [p1, p2, p3, np.asarray([mx, my])]


def curve_angle(x, y):
    if math.floor(x[1]) != math.floor(x[0]):
        a1 = math.degrees(math.atan((y[0][0] - y[1][0]) / (math.floor(x[0]) - math.floor(x[1]))))
    else:
        a1 = 90
    if math.floor(x[-1]) != math.floor(x[-2]):
        a2 = math.degrees(math.atan((y[-1][0] - y[-2][0]) / (math.floor(x[-1]) - math.floor(x[-2]))))
    else:
        a2 = 90

    if a1 <= 0:
        a1 += 180
    if a2 <= 0:
        a2 += 180

    return a1 - a2


def rotate_points(points, angle, center):
    rad = math.radians(angle)
    points = [(point[0] - math.floor(center[0]), point[1] - math.floor(center[1])) for point in points]
    points = [(math.cos(rad) * x - math.sin(rad) * y, math.sin(rad) * x + math.cos(rad) * y) for x, y in points]
    points = [(point[0] + math.floor(center[0]), point[1] + math.floor(center[1])) for point in points]
    points = [[math.floor(point[0]), math.floor(point[1])] for point in points]
    return points


def get_width_cucumber(height, mask):
    height = np.array(height)
    height = height[np.lexsort([height[:, 0], height[:, 1]])]
    point = int(len(height) / 3)  #기준: 이미지 상 위쪽 기준 위에서 1/3 지점
    points = np.array([])
    if height[point][1] == height[point - 1][1]:  # 수직일 경우
        y_all = np.unique(np.where(mask == 255)[0]).astype(np.int32)
        for i in y_all:
            if mask[i][height[point][0]] == 255:
                points = np.append(points, i)
        ret_points = [[int(height[point][0]), int(points.max())], [int(height[point][0]), int(points.min())]]
        return ret_points
    else:
        slope = -(height[point][0] - height[point - 1][0]) / (height[point][1] - height[point - 1][1])
        if abs(slope) <= 1:  # 기울기가 작을 경우
            x_all = np.unique(np.where(mask == 255)[1]).astype(np.int32)
            for i in x_all:
                if math.floor(slope * (i - height[point][0]) + height[point][1]) not in range(mask.shape[0]):
                    continue
                if mask[math.floor(slope * (i - height[point][0]) + height[point][1])][i] == 255:
                    points = np.append(points, i)
            ret_points = [[int(points.max()), math.floor(slope * (points.max() - height[point][0]) + height[point][1])], [int(points.min()), math.floor(slope * (points.min() - height[point][0]) + height[point][1])]]
        else:  # 기울기가 클 경우
            y_all = np.unique(np.where(mask == 255)[0]).astype(np.int32)
            for i in y_all:
                if math.floor((i - height[point][1]) / slope + height[point][0]) not in range(mask.shape[1]):
                    continue
                if mask[i][math.floor((i - height[point][1]) / slope + height[point][0])] == 255:
                    points = np.append(points, i)
            ret_points = [[math.floor((points.max() - height[point][1]) / slope + height[point][0]), int(points.max())], [math.floor((points.min() - height[point][1]) / slope + height[point][0]), int(points.min())]]

    return ret_points

def delete_duplicate_values_stem(bbox, bbox_list):
    ### stem의 두개의 bbox의 x min, y min, x max, y max값 중 하나라도 같은 경우 
    ### 완전히 같은 bbox로 인식하는 오류 해결을 위한 함수
    x_min, y_min, x_max, y_max = bbox
    continue_flag = False
    for box in bbox_list:  # 완전히 같은 bbox는 제외
        if x_min - box[0] == 0 and y_min - box[1] == 0 and x_max - box[2] == 0 and y_max - box[3] == 0:
            continue_flag = True

    return continue_flag