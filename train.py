from ipaddress import ip_address
from tqdm import tqdm
import sys, math
import torch
import numpy as np
print(torch.__version__, torch.cuda.is_available())

assert torch.cuda.is_available()
assert torch.__version__.startswith("1.12")


import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries

import cv2, os, random
import argparse
# from google.colab.patches import cv2_imshow       # in colab


# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode


from config import TrainConfig      # custom config
import utils as c_utl


def set_cfg_for_training(cfg, c_cfg, args, model_save_path, class_num, total_images_num):
    cfg.OUTPUT_DIR = model_save_path

    # model_zoo.get_config_file : config file을 load한다.
    # merge_from_file : 인자로 받은 config file을 existing config file과 marge한다. 
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/for_train.yaml")) # 가능
  
    if c_cfg.CONTINUE_LEARNING: 
        print(f"#      Continue training, model name : {c_cfg.CONTINUE_MODEL_NAME}")
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, c_cfg.CONTINUE_MODEL_NAME)
    else :
        # fine tuning을 위해
        # model_zoo.get_checkpoint_url : model의 path를 return
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo

    cfg.DATASETS.TRAIN = (c_cfg.DATASET_NAME + "_train", )
    cfg.DATASETS.TEST = () # (c_cfg.DATASET_NAME + "_val")
    # cfg.TEST.EVAL_PERIOD = c_cfg.EVAL_PERIOD

    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.SOLVER.IMS_PER_BATCH = c_cfg.BATCH_SIZE
    cfg.SOLVER.BASE_LR = c_cfg.LEARNING_LEATE  # pick a good LR

    if c_cfg.COMPUTE_ITER:
        cfg.SOLVER.NUM_GPUS = 1
        single_iteration = cfg.SOLVER.NUM_GPUS * cfg.SOLVER.IMS_PER_BATCH
        ITERS_IN_ONE_EPOCH = total_images_num / (single_iteration)
        cfg.SOLVER.MAX_ITER = int(ITERS_IN_ONE_EPOCH*c_cfg.EPOCH)   # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
        print(f"#    COMPUTE_ITER is True ")
    else :
        cfg.SOLVER.MAX_ITER = args.iter
    print(f"#            computed iteration : {cfg.SOLVER.MAX_ITER}")


    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = class_num 
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = c_cfg.SCORE_THRESHOLD   # set a custom testing threshold


def training(c_cfg, args):
    path_val = os.path.join(os.getcwd(), c_cfg.PATH_VAL_RESULT)
    os.makedirs(path_val, exist_ok = True)
    model_save_path = os.path.join(os.getcwd(), c_cfg.MODELS_DIR)

    dataset_metadata, class_num, val_dataset, total_images_num = c_utl.get_metadata_dataset_for_train(c_cfg, args)
  
    cfg = get_cfg()
    if not torch.cuda.is_available(): cfg.MODEL.DEVICE = "cpu"
    set_cfg_for_training(cfg, c_cfg, args, model_save_path, class_num, total_images_num)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # train and save model
    trainer = DefaultTrainer(cfg)

    trainer.resume_or_load(resume=False)
    trainer.train()


    # Inference should use the config with parameters that are used in training
    # TODO : do validation every time of save model during training 
    # validation
    # predictor = DefaultPredictor(cfg)
        
    

    
if __name__ == '__main__':
    # python train.py --type tomato --iter 200000
    # python batch_inference.py --type tomato --model 2022-08-18_1816_tomato_200000.pth
    import argparse
    c_cfg = TrainConfig

    parser = argparse.ArgumentParser(
        description="decide what action to take"
    )
    parser.add_argument(
        "--iter",
        help=" iteration number. ", default= c_cfg.ITER ,
        type=int,
    )
    parser.add_argument(
        "--type",               # paprika, melon, cucumber, onion, strawberry
        help="type of fruit", default= None ,
        type=str,
    )

    args = parser.parse_args()
    
    if args.type is None:
        print("do enter fruit type")
        sys.exit()

    training(c_cfg, args)

