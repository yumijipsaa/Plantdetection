import os, sys, cv2, json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import argparse

import torch

from config import EvaluationConfig      # custom config
import utils as c_utl

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer

# if not torch.cuda.is_available(): cfg.MODEL.DEVICE = "cpu"
assert torch.cuda.is_available()
#assert torch.__version__.startswith("1.12")


def set_cfg_for_evaluation(cfg, c_cfg, args, saved_model_path, class_num):
    cfg.OUTPUT_DIR = saved_model_path
    
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/for_train.yaml"))
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, args.model)
    cfg.DATASETS.TRAIN = (c_cfg.DATASET_NAME + "_train", )
    cfg.DATASETS.TEST = ()

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = class_num 
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = c_cfg.SCORE_THRESHOLD   # set a custom testing threshold

    cfg.DATALOADER.NUM_WORKERS = 2

    return cfg
    


def parse_gt_infer_data(masks, boxes, class_name, val_dict, classes_name):
    infer_data = []
    gt_data = []
    
    for mask, bbox, name_score in zip(masks, boxes, class_name):
        infer_dict = {}
        infer_dict['mask'] = mask.polygons      # [array([ int, int, int, ..., int, int, int], dtype=int32)]
        infer_dict['bbox'] = bbox               # [float float float float]
        
        class_name, score = name_score.split(' ')[0], name_score.split(' ')[1].split('%')[0]
        infer_dict['class_name'] = class_name             # label
        infer_dict['confidence'] = int(score)                   # int%
        
        infer_data.append(infer_dict)
    
    # ['bbox'], ['category_id'], ['segmentation'], ['bbox_mode']
    label_data = val_dict['annotations']
    for data in label_data:
        gt_dict = {}
        gt_dict['bbox'] = data['bbox']
        gt_dict['class_name'] = classes_name[data['category_id']]
        gt_data.append(gt_dict)
    
    return infer_data, gt_data


def compute_mAP(class_names_gt, threshold, infer_data, gt_data):
    assert len(threshold) ==2
    num_thrshd_divi = 9
    thrshd_value = (threshold[-1] - threshold[0]) / num_thrshd_divi
    iou_threshold_range = [round(threshold[0] + (thrshd_value*i), 2) for i in range(num_thrshd_divi+1)]

    confusion_dict = dict()
    for class_name in class_names_gt:
        confusion_dict[class_name] = []
        for iou_threshold in iou_threshold_range:
            confusion_dict[class_name].append(dict(iou_threshold = iou_threshold,
                                               num_gt = 0,
                                               num_pred = 0,
                                               num_true = 0))
            
    # for img_path_list, val_dict_list in tqdm(zip(batch_images_list, batch_val_list), total=len(batch_images_list)):      
    #     outputs, original_images  = predictor(img_path_list, c_cfg)  
    #     for output, im, path_, val_dict in zip(outputs, original_images, img_path_list, val_dict_list): \
    #         masks, boxes, class_name, out_infer = visualizer_infer.draw_instance_predictions(output["instances"].to("cpu"))
    #         if masks is None and boxes is None and class_name is None:
    #             continue
    
    
    for iou_idx, iou_threshold in enumerate(iou_threshold_range):
        for infer_idx, infer_ in enumerate(infer_data):   # 각각의 inferenced instance에 대해           
            infer_box = infer_['bbox']
            infer_class_name =  infer_['class_name'].split(" ")[0]
                        
            for gt_ in gt_data:     # 각각의 ground truth instance에 대해                
                gt_box = gt_['bbox']        # x_min, x_max, y_min, y_max
                gt_class_name = gt_['class_name']
                
                if infer_idx == 0:
                    confusion_dict[gt_class_name][iou_idx]['num_gt'] +=1
                
                
                iou = c_utl.comfute_iou(infer_box, gt_box)
                print(f"iou ; {iou}             infer_box : {infer_box}, gt_box : {gt_box}")
                
                if iou >= iou_threshold:
                    confusion_dict[infer_class_name][iou_idx]['num_pred'] +=1
                    
                if infer_class_name == gt_class_name:
                    confusion_dict[gt_class_name][iou_idx]['num_true'] +=1
  
    for class_name, iou_threshold_list in confusion_dict.items():
        for iou_idx, iou_threshold in enumerate(iou_threshold_list):
            print(f"class_name : {class_name},  iou:{iou_threshold['iou_threshold']},   num_pred:{iou_threshold['num_pred']}, num_gt:{iou_threshold['num_gt']}")
            recall =  iou_threshold['num_true']/iou_threshold['num_gt']
            
            if iou_threshold['num_pred'] ==0:
                precision = 0
            else:
                precision = iou_threshold['num_true']/iou_threshold['num_pred']
            
            confusion_dict[class_name][iou_idx]['recall'] = recall
            confusion_dict[class_name][iou_idx]['precision'] = precision
            # print(f"{class_name}, iou:{iou_threshold['iou_threshold']},     recall :{recall}, precision:{precision}")
    
    
    exit()

def make_precision_recall_list(infer_data, gt_data, class_name_dict, num_gt_class, result_img_infer):      
    tmp_tp = 0
    tmp_fp = 0
    for i, infer_ in enumerate(infer_data):   # 각각의 inferenced instance에 대해
        infer_box = infer_['bbox']
        infer_class_name =  infer_['class_name'].split(" ")[0]


        tp_or_fp = 'fp'
        ckeck_flag = True
        max_iou = 0.0
        tmp_iou_list = []
        
        for gt_ in gt_data:     # 각각의 ground truth instance에 대해
            gt_box = gt_['bbox']        # x_min, x_max, y_min, y_max
            gt_class_name = gt_['class_name']

            # if i == 0:
            #     gt_left_top, gt_right_bottom =  (int(gt_box[0]), int(gt_box[1])), (int(gt_box[2]), int(gt_box[3]))
            #     cv2.rectangle(tmp_result_img_infer, gt_left_top, gt_right_bottom, color = (0, 0, 255), thickness = 2)
        
            if i == 0:
                num_gt_class[f"{gt_class_name}"] +=1
            
            # midrid 와 leaf는 iou가 높은 값으로 겹치는 경우가 많음 (cap과 fruit 마찬가지)
            if (gt_class_name == 'leaf' and infer_class_name == 'midrid') or (infer_class_name == 'leaf' and gt_class_name == 'midrid') : continue
            if (gt_class_name == 'cap' and infer_class_name == 'fruit') or (infer_class_name == 'fruit' and gt_class_name == 'cap') : continue
            

            ## comfute iou     
            iou = c_utl.comfute_iou(infer_box, gt_box)  
            if iou == 0.0: continue   
            
            tmp_iou_list.append(iou)
            # inferenced instance 가 각각의 ground truth instance에 대해 iou를 계산 후, 계산 된 iou중 max value iou를 가져온다.
            if max_iou < iou:   
                max_iou = iou
                gt_name, infer_name = gt_class_name, infer_class_name
                ckeck_flag = False

                tmp_gt_box = gt_box
                tmp_gt_class = gt_class_name

        if ckeck_flag:      # inferenced instance가 의미없는 instance일 경우 (모든 GT box에도 전혀 intersection이 없는 경우)
            confidence_score = 0.0
            tp_or_fp = 'fp'
            tmp_fp +=1
        else:

            if max_iou >= 0.5 :     # and iou <= 0.95:  # 0.95보다 높은 iou가 많음    
                if gt_name == infer_name : 
                    confidence_score = infer_['confidence'] * max_iou
                    tp_or_fp = 'tp'
                    tmp_tp +=1
                else : 
                    confidence_score = infer_['confidence'] * max_iou
                    tp_or_fp = 'fp'
                    tmp_fp +=1
            else :      # iou가 낮은경우
                ##  labeling 과정에서 GT로 그리지 않은 instance를 detection하는 경우 
                #   1. labeling 대상인데 labeling을 하지 않았지만, inference과정에서 detecting된 경우 
                #   2. inference과정에서 올바르지 않은 intance를 detection했는데, 우연히 임의의 GT box와의 intersection이 존재하는 경우
                confidence_score = infer_['confidence'] * max_iou
                tp_or_fp = 'fp'
                tmp_fp +=1

                # print(f"predicted bbox: {infer_box}")   
                # print(f"groundtruth bbox: {tmp_gt_box}")   
                # print(f"iou: {max_iou}")   
                # infer_left_top_point, infer_right_bottom_point =  (int(infer_box[0]), int(infer_box[1])), (int(infer_box[2]), int(infer_box[3]))
                # print(f"    연두색 infer_left_top_point, infer_right_bottom_point: {infer_left_top_point, infer_right_bottom_point}")   
                # cv2.rectangle(result_img_infer, infer_left_top_point, infer_right_bottom_point, color = (0, 255, 0), thickness = 2)

                # gt_left_top_point, gt_right_bottom_point =  (int(tmp_gt_box[0]), int(tmp_gt_box[1])), (int(tmp_gt_box[2]), int(tmp_gt_box[3]))
                # cv2.rectangle(result_img_infer, gt_left_top_point, gt_right_bottom_point, color = (255, 0, 0), thickness = 2)
                # print(f"    파란색 gt_left_top_point, gt_right_bottom_point: {gt_left_top_point, gt_right_bottom_point}") 
                # print(f"   연두색 infer_class_name : {infer_class_name}        파란색 gt_class_name : {tmp_gt_class}" ) 
                # print(f" tmp_iou_list : {tmp_iou_list[-1]} \n")


                # cv2.imshow("tmp", result_img_infer)
                # while True:
                #     if cv2.waitKey() == 27:
                #         break
                

        # 모든 gt에 대해 iou가 0일 경우 tp_or_fp == 'fp'
        # tp_or_fp : 'tp' or 'fp'
        class_name_dict[f"{infer_class_name}"].append([confidence_score, tp_or_fp])

    # 한 개의 image에서 fp와 tf의 개수를 보여준다.
    # print(f"    tmp_fp : {tmp_fp}, tmp_tp : {tmp_tp}")

    return class_name_dict, num_gt_class, result_img_infer
    

# class 1개에 대한 AP계산
def compute_average_precision(precision_recall_list, class_name, plt_save_path):
    # TODO : fix compute function to add all area between all of each point  
    precision_recall_list.sort(key=lambda x: x[1])  # recall 기준으로 sort
    area_list = []

    
    start_vertical_flag = True  # 수직 상승이 시작되기 전엔 True
    continue_vertical_flag = False   # 수직 상승인 도중엔 True
    reverse_precision_recall_list = precision_recall_list[::-1]
    max_precision, before_recall, bottom_length, i = None, None, None, 0
    

    precision_list, recall_list = [], []
    tmp_precision_list, tmp_recall_list = [], []
    for precision, recall in reverse_precision_recall_list:         # precision-recall curve의 우측에서 좌측방향으로 탐색
        i +=1
        
        # print(f"    num : {i}")
        # print(f"precision : {precision}, recall : {recall}")
        # scatter을 위한 list
        precision_list.append(precision)    
        recall_list.append(recall)

        if before_recall is None:           # initial point
            before_recall = recall
            max_precision = precision
            bottom_length = 0
            tmp_precision = precision
            tmp_recall = recall
            continue
    
        if recall == before_recall:     # 수직 상승일 때
            if start_vertical_flag :     # 수직상승이 시작됐을 때
                area = bottom_length * max_precision
                area_list.append(area)
                # print(f"    수직 상승 등장, bottom_length : {bottom_length}, max_precision : {max_precision}, area : {area}")
                max_precision = precision
                start_vertical_flag = False
                continue_vertical_flag = True
                bottom_length = 0

                tmp_precision_list.append(tmp_precision)
                tmp_recall_list.append(tmp_recall)
                tmp_precision = precision
                tmp_recall = recall
                
            else:               # 수직상승이 한번 더 이어서 나왔을 때
                continue_vertical_flag = True
                max_precision = precision
                tmp_precision = precision
                tmp_recall = recall
                # print(f" 이어지는 수직 상승")
            
            
        elif max_precision >= precision:      # 좌 하향 또는 좌 평행일 때
            if i == len(reverse_precision_recall_list):     # last point
                bottom_length += before_recall
                area = bottom_length * max_precision
                area_list.append(area)                      # 바로 이전 point에 대한 area계산
                # print(f" 좌 하향 또는 좌 평행 last_point bottom_length : {bottom_length}, max_precision : {max_precision}, area : {area}")

                tmp_precision_list.append(tmp_precision)
                tmp_recall_list.append(tmp_recall)
                continue
            
            bottom_length += (before_recall - recall)   # 하단 길이 축적
            before_recall = recall
            # print(f" 좌 하향 또는 좌 평행 bottom_length : {bottom_length}")
            start_vertical_flag = True

            if continue_vertical_flag :     # 수직상승이 끝난 직후
                continue_vertical_flag = False
                tmp_precision_list.append(tmp_precision)
                tmp_recall_list.append(tmp_recall)            

        else:     # 좌 상향 일 때 (없음)
            pass


    average_precision_dict = {}
    average_precision_dict["average_precision"] = 0
    average_precision_dict["class_name"] = class_name
    for area in area_list:
        average_precision_dict["average_precision"] += area
    # print(f"    class_name : {class_name}")
    # print(f"    average_precision : {average_precision}")

    plt.scatter(recall_list, precision_list, color='darkmagenta')
    plt.scatter(tmp_recall_list, tmp_precision_list, marker='^', color='limegreen')
    plt.title(f"{class_name}, average_precision = {average_precision_dict['average_precision']:.4f}")

    plt.savefig(os.path.join(plt_save_path,  f"{class_name}.jpg"))
    plt.clf()

    
    return average_precision_dict


def comfute_mAP(class_name_dict, num_gt_class, plt_save_path):

    average_precision_dict_list = []

    class_num = len(list(class_name_dict.keys()))
    total_average_precision = 0
    for class_name, class_name_num in zip(list(class_name_dict.keys()), list(num_gt_class.keys())):
        class_name_dict[f'{class_name}'].sort(reverse=True)     # confidence_score기준으로 sort
        
        precision_recall_list = []

        fp_count , tp_count = 0, 0
        for score_and_tpfp in class_name_dict[f"{class_name}"]:

            if score_and_tpfp[1] == 'fp':
                fp_count +=1
                
            elif score_and_tpfp[1] == 'tp':
                tp_count +=1

            # precision_recall_list[0] : [precision, recall]
            precision = tp_count / (fp_count + tp_count)
            recall = tp_count / num_gt_class[class_name_num]
            precision_recall_list.append([precision,recall])
            # print(f" precision : {precision:.3f} , recall : {recall:.3f}")

        # print(f"    tp_count : {tp_count}, fp_count : {fp_count}, num_gt_class : {num_gt_class[f'{class_name_num}']}")
        
        average_precision_dict = compute_average_precision(precision_recall_list, class_name, plt_save_path)     
        total_average_precision += average_precision_dict["average_precision"]
        average_precision_dict_list.append(average_precision_dict)

    mAP = total_average_precision/class_num
    return mAP, average_precision_dict_list


def save_summary_text_file(c_cfg, args, plt_save_path, val_dataset, average_precision_dict_list, mAP):
    
    
    validation_dataset_path = os.path.join(os.path.join(os.path.join(os.getcwd(), c_cfg.TRAIN_DIR), args.type), c_cfg.TRAIN_DATASET)
    summary_file_path = os.path.join(plt_save_path, "summary.txt")
    summary_file = open(summary_file_path, 'w')
    summary_file.write(f"model name : {args.model} \n\n")
    summary_file.write(f"validation dataset path : {validation_dataset_path} \n")
    summary_file.write(f"number of validation data = {len(val_dataset)} \n \n")

    summary_file.write(f"number of class = {len(average_precision_dict_list)} \n")

    for average_precision_dict in average_precision_dict_list:
        summary_file.write(f"average precision of {average_precision_dict['class_name']} = {average_precision_dict['average_precision']:.3f} \n")
    summary_file.write(f"mAP = {mAP:.3f} \n")
    summary_file.close()




# 특정 train dataset으로부터 validation data를 parsing한 후 원하는 model을 통해 evaluation을 진행한다.
def evalutation(args, c_cfg):
    
    # set config and gat dataset for evaluation
    cfg = get_cfg()
    saved_model_path = os.path.join(os.getcwd(), c_cfg.MODELS_DIR)
    c_cfg.TRAIN_EVALUATION_RATIO = args.ratio
    
    metadata_infer, metadata_gt, class_num, val_dataset, class_names_gt = c_utl.get_metadata_dataset_for_evaluation(c_cfg, args)
    cfg = set_cfg_for_evaluation(cfg, c_cfg, args, saved_model_path, class_num)

    path_eval = os.path.join(os.getcwd(), c_cfg.PATH_EVAL_RESULT)
    os.makedirs(path_eval, exist_ok = True)
    type_dir_path = os.path.join(path_eval, args.type)
    os.makedirs(type_dir_path, exist_ok = True)

    model_evaludat_result_path = os.path.join(type_dir_path, f"{args.model.split('.')[0]}_{args.ratio}")  
    os.makedirs(model_evaludat_result_path, exist_ok = True)
    summary_save_path = os.path.join(model_evaludat_result_path, "summary")  
    os.makedirs(summary_save_path, exist_ok = True)

    # evaluation을 위한 predict instance 생성
    predictor = DefaultPredictor(cfg)

    image_path_list = []
    img_path_list = []
    
    for val_data in val_dataset:
        image_path_list.append(val_data['file_name'])

    batch_images_list = [image_path_list[x:x + c_cfg.EVALUATION_BATCH_SIZE] for x in range(0, len(image_path_list), c_cfg.EVALUATION_BATCH_SIZE)]
    batch_val_list = [val_dataset[x:x + c_cfg.EVALUATION_BATCH_SIZE] for x in range(0, len(val_dataset), c_cfg.EVALUATION_BATCH_SIZE)]

    class_name_dict = {}
    for class_name in class_names_gt:
        class_name_dict[f"{class_name}"] = []

    num_gt_class = {}
    for class_name in class_names_gt:
        num_gt_class[f"{class_name}"] = 0

    for img_path_list, val_dict_list in tqdm(zip(batch_images_list, batch_val_list), total=len(batch_images_list)):      
        print(f"img_path_list : {len(img_path_list)}")
        outputs, original_images  = predictor(img_path_list, c_cfg)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format

        for output, im, path_, val_dict in zip(outputs, original_images, img_path_list, val_dict_list): # batch로부터 1개씩 

            visualizer_infer = Visualizer(im[:, :, ::-1],
                        metadata=metadata_infer,
                        scale=1,
                        instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
            )

            # image 1개에 대한 mask, bbox, label
            masks, boxes, class_name, out_infer = visualizer_infer.draw_instance_predictions(output["instances"].to("cpu"))

            if masks is None and boxes is None and class_name is None:
                continue
            
            img_label = cv2.imread(val_dict["file_name"])
            visualizer_gt = Visualizer(img_label[:, :, ::-1], metadata=metadata_gt, scale=1)
            out_gt = visualizer_gt.draw_dataset_dict(val_dict)
            result_img_label = out_gt.get_image()[:, :, ::-1]

            result_img_infer = np.array(out_infer.get_image()[:, :, ::-1])
                
            infer_data, gt_data = parse_gt_infer_data(masks, boxes, class_name, val_dict, class_names_gt)
            # compute_mAP(class_names_gt, c_cfg.THRESHOLD, infer_data, gt_data)
            class_name_dict, num_gt_class, result_img_infer = make_precision_recall_list(infer_data, gt_data, class_name_dict, num_gt_class, result_img_infer)
            
            
            # cv2.imshow("infer", result_img_infer)
            # while True:
            #     if cv2.waitKey() == 27:
            #         break
            if c_cfg.SAVE_FLAG : 
                if os.name == 'nt':
                    file_name = path_.split('\\')
                elif os.name == 'posix':
                    file_name = path_.split('/')


                file_name_label = file_name[-1].split(".")[0] + "_label_." + file_name[-1].split(".")[-1]
                save_path = os.path.join(model_evaludat_result_path, file_name_label)
                cv2.imwrite(save_path, result_img_label)

                file_name_infer = file_name[-1].split(".")[0] + "_infer_." + file_name[-1].split(".")[-1]
                save_path = os.path.join(model_evaludat_result_path, file_name_infer)
                cv2.imwrite(save_path, result_img_infer)
    
    mAP, average_precision_dict_list = comfute_mAP(class_name_dict, num_gt_class, summary_save_path)
    save_summary_text_file(c_cfg, args, summary_save_path, val_dataset, average_precision_dict_list, mAP)


if __name__ == '__main__':
    c_cfg = EvaluationConfig

    ## python evaluation.py --model 2022-10-19_194_paprika_275000.pth --type paprika --ratio 0.9
    parser = argparse.ArgumentParser(
        description="decide what action to take"
    )
    parser.add_argument(
        "--model",
        help=" model file name", default= None,
        type=str,
    )
    parser.add_argument(
        "--type",               # paprika, melon, cucumber, onion, strawberry
        help="type of fruit", default= None ,
        type=str,
    )
    parser.add_argument(
        "--ratio",
        help= "split ratio from train dataset for evaluation", default=c_cfg.TRAIN_EVALUATION_RATIO,
        type=float
    )
    
    parser.add_argument(
        "--json", type=str
    )
    
    
    args = parser.parse_args()

    if args.type is None or args.model is None:
        print("do enter --type or --model option")
        sys.exit()

    evalutation(args, c_cfg)

