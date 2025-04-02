import os, math
import argparse
import json
from re import T
import sys, time

from labelme import utils
import numpy as np
import glob
import PIL.Image
import cv2
from tqdm import tqdm

import detectron2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode


# 통합 dataset의 file name
DEFAULT_DATASET_NAME = "train_dataset.json"


# 통합 json file을 save할 dir의 name
DIR_NAME_DATASET = "train_dataset" 

# True이면 labeling된 image의 원본을 따로 저장
SAVE_ORIGINAL_IMAGE = False
# labeling된 image의 원본을 저장할 directory의 name
DIR_NAME_SELECTED_IMAGE = "original_images"

# True이면 GT image를 save
SAVE_GT_IMAGE = False
# mask가 그려진 GT image를 저장할 directory의 name
DIR_NAME_VISUALIZE_IMAGE = "visualizing"

# True이면 통합 json파일에 original image까지 mapping해서 save  
# (저장속도 느림, training시 original image를 따로 directory에 위치시켜야 함.)
# False이면 json파일에 original image를 mapping하지 않고 save
MAPPING_ORIGINAL_IMAGE = False

# agumentation
VERTICAL_FLIP_IMAGE = False      # vertical flip
RESIZE_IMAGE = False             # resizing 
MAX_SIZE = 1280

# 하용되는 object이름
ORIGINAL_CLASS_NAME_LIST = [['leaf'], ['petiole'], ['stem'], ['fruit'], ['cap'],['growing'],['cap_2'],['node'],['midrib'],['bud_flower'],['length'],['y_fruit'],['flower'],
                         ['scion_length'],['rootstock_length']]
ONLY_NON_POINT = True   # point 라벨링이 되어있는 dataset이 포함되어 있지만, point label을 사용하지 않을 경우 True

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def get_dataset_dicts(dataset, image_dir_path):
    train_dataset_dicts_list = []

    c = 0
    for idx, metadata in enumerate(dataset["metadatas"]):
        record = {}
        image_info = metadata["image_info"]
        file_name = image_info["file_name"]    # 2_20200825_093215.jpg
        file_path = os.path.join(image_dir_path, file_name)     # training에 사용할 image의 path    

        record["file_name"] = file_path                  # file의 path(name아님)
        record["image_id"] = image_info["image_id"]
        record["height"] = image_info["height"]
        record["width"] = image_info["width"]
        
        objs = []
  
        for _, instance_info in enumerate(metadata["instance_info"]):
            
            if list(instance_info.keys()) == ['segmentation', 'iscrowd', 'area', 'bbox', 'category_id', 'instance_id'] \
                or list(instance_info.keys()) == ['class_name', 'segmentation', 'iscrowd', 'area', 'bbox', 'category_id', 'instance_id']:
    
                x_min, y_min, x_val, y_val = instance_info["bbox"]      # x_max = x_min + x_val (same y)
                obj = {
                    "bbox":  [x_min, y_min, x_min + x_val, y_min + y_val],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": instance_info["segmentation"],
                    "category_id": instance_info["category_id"],
                    }
                objs.append(obj)
                c +=1  
            else : pass
            

        record["annotations"] = objs 

        train_dataset_dicts_list.append(record)

    return train_dataset_dicts_list



def get_class_name(dataset):
    class_name = dataset["classes"]
    return class_name


        

class labelme2coco(object):
    def __init__(self, args):
        """
        :param labelme_json: the list of all labelme json file paths
        :param save_json_path: the path to save new json
        """
        image_dir_path = os.path.join(os.getcwd(), args.labelme_images)
        labelme_json = glob.glob(os.path.join(image_dir_path, "*.json"))
        self.args = args

        self.labelme_json = labelme_json
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0

        self.image_dir = None
        self.dataset_dir =  None
        self.set_dir()

        self.classes = []
        self.metadatas = []
        self.dataset = {}
        self.do_augmentation_h_flip = False        # vertical flap image를 따로 추가로 저장하는 경우 True (augmenatation)   VERTICAL_FLIP_IMAGE 로 조정
        self.do_augmentation_resizing = False      # resized image를 따로 추가로 저장하는 경우 True (augmenatation)         RESIZE_IMAGE 로 조정
        self.do_augmentation_h_flip_resizing = False
        self.ratio = 0

        self.save_flag = True   # False이면 image 저장 및 metadataset에 포함 안함

        self.save_json()


    def set_dir(self):
        traindataset_dir_path = os.path.join(os.getcwd(), DIR_NAME_DATASET)
        os.makedirs(traindataset_dir_path, exist_ok=True)

        yy_mm_dd = time.strftime('%Y-%m-%d', time.localtime(time.time())) +"_"+ self.args.labelme_images
        self.dataset_dir = os.path.join(traindataset_dir_path, yy_mm_dd)

        os.makedirs(self.dataset_dir, exist_ok=True)
        if SAVE_ORIGINAL_IMAGE : 
            self.image_dir = os.path.join(self.dataset_dir, DIR_NAME_SELECTED_IMAGE)
            self.imgae_dir_visual = os.path.join(self.dataset_dir, DIR_NAME_VISUALIZE_IMAGE)
            os.makedirs(self.image_dir, exist_ok=True)
            os.makedirs(self.imgae_dir_visual, exist_ok=True)



    def data_transfer(self):
        count = 0
        for json_file in tqdm((self.labelme_json)):
            self.get_metadatas(json_file, count+1, None)
            if VERTICAL_FLIP_IMAGE:
                self.do_augmentation_h_flip = True
                self.get_metadatas(json_file, count+1, None)
                self.do_augmentation_h_flip=False

            if RESIZE_IMAGE:
                # self.do_augmentation_resizing = True
                # self.get_metadatas(json_file, count+1, 1280)
                # self.do_augmentation_resizing = False

                self.do_augmentation_resizing = True
                self.get_metadatas(json_file, count+1, 960)
                self.do_augmentation_resizing = False
            
            if VERTICAL_FLIP_IMAGE and RESIZE_IMAGE:
                # self.do_augmentation_h_flip_resizing = True
                # self.get_metadatas(json_file, count+1, 1280)
                # self.do_augmentation_h_flip_resizing = False

                self.do_augmentation_h_flip_resizing = True
                self.get_metadatas(json_file, count+1, 960)
                self.do_augmentation_h_flip_resizing = False

        self.dataset["metadatas"] = self.metadatas
        self.dataset["classes"] = self.classes
        
        


    def get_metadatas(self, json_file, count, size = None):

        metadata = {}
        with open(json_file, "r") as fp:
            data = json.load(fp)     
            
            self.save_flag = True
            image_info = self.get_image_info(data, count, size)
            
            # image_path = os.path.join(os.path.join(os.getcwd(), "paprika_thiennp"), data["imagePath"])      #####
            # img = cv2.imread(image_path)
            
            if self.save_flag:

                instance_info = []

                for _, shapes in enumerate(data["shapes"]):

                    label = []
                    label.append(shapes["label"])   

                    instance = {}

                    if label not in ORIGINAL_CLASS_NAME_LIST:
                        if ONLY_NON_POINT: pass  
                        else:
                            # self.label에 이미 존재하는 label이 있는지 확인
                            if label not in self.label:     # label의 type이 list여야 함
                                self.label.append(label)   
                                self.classes.append(label[0])      
                            else :   
                                pass     
                    else :                
                        # self.label에 이미 존재하는 label이 있는지 확인
                            if label not in self.label:     # label의 type이 list여야 함
                                self.label.append(label)   
                                self.classes.append(label[0])      
                            else :   
                                pass   
                    

                    points = shapes["points"]
                    
                    if label not in ORIGINAL_CLASS_NAME_LIST:
                        if ONLY_NON_POINT: pass  
                        else : 
                            # ['points', 'category_id', 'instance_id']
                            instance = self.point_annotation(points, label, instance)
                            # print(f"label : {label},    points : {len(points)}, real_points: {len(instance['points'])}")
                    else : 
                        # ['segmentation', 'iscrowd', 'area', 'bbox', 'category_id', 'instance_id']
                        instance = self.mask_annotation(points, label, instance, image_info['file_name'])

                    if ONLY_NON_POINT:
                        if len(list(instance.keys()))==0:
                            pass
                        else :
                            self.annID += 1
                            instance_info.append(instance)

        if self.save_flag:              
            metadata["image_info"] = image_info
            metadata["instance_info"] = instance_info

            self.metadatas.append(metadata)
        

    def resizing_image(self, img, data, image_info, file_name, max_size, org_size):
        tmp_file_name = data["imagePath"].split("/")[-1]
        image_info["file_name"] = tmp_file_name.split(".")[0] + file_name + tmp_file_name.split(".")[-1]

        new_size = (max_size, max_size)      # 바꾸고자 하는 size (width, height) # width, height중 큰 값이 1280이 된다.
        img_shape = list(img.shape)   # (height, width)
        if img_shape[:2] != org_size:                 
            tall = True if img_shape[1] < img_shape[0] else False

            if tall:    # height < width
                ratio = new_size[1] / float(img_shape[0])
                new_dimension = (int(img_shape[1] * ratio), int(img_shape[0] * ratio))
            else:       # width < height 
                ratio = new_size[0] / float(img_shape[1])
                new_dimension = (int(img_shape[1] * ratio), int(img_shape[0] * ratio))    
            self.ratio = ratio  # segmentation coordinates의 위치를 바꿀때 사용하기 위해
            img = cv2.resize(img,dsize=new_dimension)
        else: self.save_flag = False        # image의 width, height가 이미 1280, 720이면 pass
        
        return img, image_info


    def get_image_info(self, data, count, size = None):
        if size is not None : 
            if size == 1280: org_size = [720, 1280]   
            elif size == 960: org_size = [540, 960]     # 1280 * 0.75 = 960
        
        image_info = {}
        img = utils.img_b64_to_arr(data["imageData"])
        self.original_img = img

        if self.do_augmentation_h_flip:        # image를 horizon flip
            tmp_file_name = data["imagePath"].split("/")[-1]
            image_info["file_name"] = tmp_file_name.split(".")[0] + "_flap_h." + tmp_file_name.split(".")[-1]
            img = cv2.flip(img, 1)
        elif self.do_augmentation_resizing:    # image를 resizing
            img, image_info = self.resizing_image(img, data, image_info, f"_resize_{size}.", size, org_size)  

        elif self.do_augmentation_h_flip_resizing:  # image를 resizing하고 horizon flip
            img, image_info = self.resizing_image(img, data, image_info, f"_flap_h_resize_{size}.", size, org_size)     
            img = cv2.flip(img, 1)

        else:
            image_info["file_name"] = data["imagePath"].split("/")[-1]


        self.height, self.width = img.shape[:2]
        image_info["height"] = self.height
        image_info["width"] = self.width
        image_info["image_id"] = count
            

        if MAPPING_ORIGINAL_IMAGE : 
            image_info["original_image"] = img
            # print(f" image name : {image_info['file_name']}")

        
        if SAVE_ORIGINAL_IMAGE and self.save_flag :   
            # img[:, :, ::-1] == img는 BRG이기 때문에 RGB로 바꿔준다.
            cv2.imwrite(self.image_dir + "\\" + image_info["file_name"], img[:, :, ::-1]) 
            # print(f" save original image : {image_info['file_name']}")
        
        self.img = img  
        img = None

        return image_info

    def point_annotation(self, point, label, instance):
        real_points = self.extract_real_point(point)
       
        instance['points'] = real_points

        for idx, class_ in enumerate(self.classes):
            if class_ == label[0]:
                instance["category_id"] = idx
                break

        instance["instance_id"] = self.annID
        return instance


    def extract_real_point(self, point):
        tmp_points = []

        break_flag = False
        for i, p_1 in enumerate(point):  
            if break_flag :
                break  

            for j in range(i+1, len(point)):
                p_2 = point[j]
                length_1 = math.sqrt(math.pow(p_1[0] - p_2[0], 2) + math.pow(p_1[1] - p_2[1], 2))

                if length_1 < 2.5:
                    tmp_points.append(p_1)
                    break_flag = True
                    break
        
        real_points = []
        edge_point = False
        break_flag = False
        y_sorted_point = point.copy()
        y_sorted_point.sort(key = lambda x:x[1])

        
        if len(tmp_points) == 0:
            real_points = point
        else:
            for i, p_1 in enumerate(point):  
                if break_flag:
                    break

                if i == 0:
                    real_points.append(p_1)
                    # print(f"append : {i+1}")
                    continue

                for j in range(i+1, len(point)):
                    p_2 = point[j]
                    length_1 = math.sqrt(math.pow(p_1[0] - p_2[0], 2) + math.pow(p_1[1] - p_2[1], 2))
                    

                    if length_1 < 2.5:
                        real_points.append(p_1)
                        edge_point = True
                        # print(f"append : {i+1}")
                        break

                    if j == len(point) - 1 and edge_point:
                        real_points.append(p_1)                    
                        edge_point = False    
                        # print(f"append : {i+1}")
                        break


                    if p_1 == y_sorted_point[-1]:
                        real_points.append(p_1)
                        break_flag = True
                        break
        return real_points

    def mask_annotation(self, points, label, instance, file_name):
        contour = np.array(points)

        if self.do_augmentation_h_flip:
            for i, con in enumerate(contour):   # segmentation 좌표들을 좌우 대칭
                contour[i, 0] = self.width - con[0]

        elif self.do_augmentation_resizing:    
            for i, con in enumerate(contour):   # segmentation 좌표들에 resizing ratio 곱
                contour[i, 0], contour[i, 1] = con[0]*self.ratio, con[1]*self.ratio
    
        elif self.do_augmentation_h_flip_resizing:
            for i, con in enumerate(contour):   # segmentation 좌표들에 resizing ratio 곱
                contour[i, 0], contour[i, 1] = con[0]*self.ratio, con[1]*self.ratio
            for i, con in enumerate(contour):   # segmentation 좌표들을 좌우 대칭
                contour[i, 0] = self.width - con[0]
        
        
        instance["segmentation"] = [list(np.asarray(contour).flatten())]

        x = contour[:, 0]
        y = contour[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        # assert area > 10, f"{file_name},   area : {area}"
        
        mask = self.polygons_to_mask([self.height, self.width], contour)
             

        # instance["iscrowd"]
        # 1인 경우 == instance가 여러 object의 집합인 경우
        # 0인 경우 == instance가 단일 object인 경우 
        instance["iscrowd"] = 0  
        instance["area"] = float(area)
        instance["bbox"] = list(map(float, self.getbbox(mask)))    #  이 line이 바로 위 instance["iscrowd"], instance["area"]보다 먼저 선언되면 visualizing안됨

        for idx, class_ in enumerate(self.classes):
            if class_ == label[0]:
                instance["category_id"] = idx
                break

        instance["instance_id"] = self.annID
        return instance


    def getbbox(self, mask):
        return self.mask2box(mask)

    def mask2box(self, mask):
        index = np.argwhere(mask == 1)   
        
        rows = index[:, 0]
        clos = index[:, 1]       

        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        
        return [
            left_top_c,
            left_top_r,
            right_bottom_c - left_top_c,
            right_bottom_r - left_top_r,
        ]

    def polygons_to_mask(self, img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)
        
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))  
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask


    def draw_save_GT_image(self):
        MetadataCatalog.get("ITC").set(thing_classes=get_class_name(self.dataset))
        GT_metadata = MetadataCatalog.get("ITC")
        dataset_dicts = get_dataset_dicts(self.dataset, self.image_dir)

        for d in tqdm(dataset_dicts):
            file_path = d["file_name"].split('\\')
            file_name = file_path[-1]

            
            img = cv2.imread(d["file_name"],  cv2.IMREAD_UNCHANGED)
            
            visualizer = Visualizer(img[:, :, ::-1], metadata=GT_metadata, scale=1)
            out = visualizer.draw_dataset_dict(d)
            image = out.get_image()[:, :, ::-1]
            
            cv2.imwrite(self.imgae_dir_visual + "\\" + file_name, image)  
            

    def save_json(self): 
        self.data_transfer()
        
        # save DEFAULT_DATASET_NAME
        print(f"name to save is : {DEFAULT_DATASET_NAME} \n")
        save_path = self.dataset_dir + "\\" +  DEFAULT_DATASET_NAME
        
        # leaf_continue_count = 10000
        # dataset = dict()
        # for key, value in self.dataset.items():
        #     if key =='classes':
        #         dataset[key] = value
        #     else:
        #         dataset[key] = []
        #         for image_meta in value:
        #             image_meta_dict = dict()
        #             for meta_key, meta_value in image_meta.items():
        #                 if meta_key=='image_info':
        #                     image_meta_dict[meta_key] = meta_value
        #                 else:
        #                     image_meta_dict[meta_key] = []
        #                     for instance in meta_value:
        #                         class_name = self.dataset['classes'][instance['category_id']]
        #                         if class_name == 'midrid':
        #                             image_meta_dict[meta_key].append(instance)
        #                         elif class_name == "leaf":
        #                             if leaf_continue_count >=0: 
        #                                 leaf_continue_count -=1
        #                                 continue
                                    
        #                         image_meta_dict[meta_key].append(instance)
        #             dataset[key].append(image_meta_dict)
                   
                   
        # self.dataset = dataset    
            
        classes_dict = dict()
        for class_name in self.dataset['classes']:
            classes_dict[class_name] = []

        for image_meta in self.dataset['metadatas']:
            for instance in image_meta['instance_info']:
                class_name = self.dataset['classes'][instance['category_id']]
                classes_dict[class_name].append(instance)
                
        for class_name in self.dataset['classes']:
            print(f"class_name : {class_name}")
            print(f"mount: {len(classes_dict[class_name])}")     
        

        json.dump(self.dataset, open(save_path, "w"), indent=4, cls=NpEncoder)
        
        
        time_now_json_name = ( time.strftime('%Y-%m-%d', time.localtime(time.time())) 
                    + "_"+ str(time.localtime(time.time()).tm_hour) 
                    + str(time.localtime(time.time()).tm_min) 
                    + "_" + self.args.type +".json")

        # save FOR_INFERENCE
        print(f"name to save is : {time_now_json_name} \n")

        save_path = os.path.join(self.dataset_dir, time_now_json_name)
        json.dump(self.dataset["classes"], open(save_path, "w"), indent=4, cls=NpEncoder)

        print("save successfully to json")

        if SAVE_GT_IMAGE:
            self.draw_save_GT_image()


if __name__ == "__main__":
    # python labelme2coco.py tomato --type tomato
    import argparse

    parser = argparse.ArgumentParser(
        description="labelme annotation to coco data json file."
    )
    parser.add_argument(
        "labelme_images",
        help="Directory to labelme images and annotation json files.",
        type=str,
    )
    parser.add_argument(
        "--type", help="type", default=DEFAULT_DATASET_NAME
    )

    args = parser.parse_args()

    image_dir_path = os.path.join(os.getcwd(), args.labelme_images)

    labelme2coco(args)
    

    
    

   
