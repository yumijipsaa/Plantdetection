# Run
import torch

torch.manual_seed(1542)
torch.cuda.manual_seed(1542)
torch.backends.deterministic = True
torch.backends.benchmark = False

print(f'pytorch version: {torch.__version__}')
print(f'Cuda is available: {torch.cuda.is_available()}')
from detectron2.utils.logger import setup_logger
setup_logger()
import cv2, os, glob
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer

from find_algorithm import Find_coordinates
from config import InferenceConfig, Find_algorithm_config
import utils as c_utl
import math 
import time ,sys
import argparse



def paprika_algorithm(find_condig, class_name_list, outputs, img, file_name, result_dir, plant_type):
    get_coordinates_segmentation = Find_coordinates(find_condig, plant_type, class_name_list)
    drawn_image, has_mask, coordinates_dict, segmentations = get_coordinates_segmentation.calculate_coordinates(outputs, img)

    if has_mask:
        if c_cfg.DRAW_COORDINATES:
            drawn_image = c_utl.draw_coordinates(coordinates_dict, drawn_image)
        if c_cfg.PRINT_COORDINATES: 
            json_dict = c_utl.save_to_json(coordinates_dict, segmentations)

        file_name_draw, file_extension = file_name.split('.')
        file_name_draw = file_name_draw + "_coordinates." + file_extension

        print(f"#            file_name_draw : {file_name_draw}")
        cv2.imwrite(os.path.join(result_dir,file_name_draw), drawn_image)
        return json_dict
    else :
        print("#             there is nothing to draw.")
        return None


def set_cfg_for_inference(cfg, c_cfg, args, model_saved_path, class_num):
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/for_train.yaml")) # 가능


    cfg.MODEL.ROI_HEADS.NUM_CLASSES = class_num  # 반드시 class의 개수와 맞아야함

    cfg.OUTPUT_DIR = model_saved_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = c_cfg.SCORE_THRESHOLD  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, args.model)




def print_time(total_time, num_images):
    if total_time >= 60 and total_time < 3600:
        print(f"# number of inferred images : {num_images},  total time : {total_time/60:.5f} minutes") 
    elif total_time >= 3600 :
        print(f"# number of inferred images : {num_images},  total time : {total_time/3660:.5f} hours") 
    else :
        print(f"# number of inferred images : {num_images},  total time : {total_time:.5f} sec")
    print("# ----- end -----")


def inference(c_cfg, args):
    
    if c_cfg.SAVE_DRAW_IMAGE : 
        find_condig = Find_algorithm_config
    test_dir = os.path.join(os.getcwd(), args.dir)
    file_list = glob.glob(test_dir + '/*.jpg')
    if len(file_list) == 0:
        print(f"# Unvalid dir path \n# dir : {args.dir}")
        sys.exit()
    
    result_dir = os.path.join(os.getcwd(), c_cfg.PATH_INFER_RESULT)
    os.makedirs(result_dir, exist_ok = True)

    model_save_path = os.path.join(os.getcwd(), c_cfg.LOAD_MODEL_DIR)
    if args.model not in os.listdir(model_save_path):
        print("# Trained model file dose not exist!")
        print(f"# path : {os.path.join(model_save_path,args.model)}")
    
    dataset_metadata, class_name_list, num_class = c_utl.get_metadata_for_inference(c_cfg, args)

    cfg = get_cfg()
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = 'cpu'

    set_cfg_for_inference(cfg, c_cfg, args, model_save_path, num_class)
   

    predictor = DefaultPredictor(cfg)

    total_time = 0
    
    # file_list = random.sample(file_list, len(file_list))
    num_images = len(file_list)
    for idx, path_ in enumerate(file_list):
        print(f"\n#----- ( {idx + 1} / {num_images}) -----")
        start = time.time()         
        math.factorial(100000) 


        img = cv2.imread(path_)
        if c_cfg.RESIZE_IMAGE:
            img = c_utl.resize_image(img)
        im = img.copy()

        outputs = predictor(im)
        
        if c_cfg.PRINT_OBJECT_INFO :
            num_instance = len(outputs["instances"].pred_classes)
            print(f"#    num of detected object : {num_instance}")
            for idx, output in enumerate(outputs["instances"].pred_classes):
                print(f"#   - object{idx + 1} : {class_name_list[output]}")

        # print(outputs["instances"].pred_classes)
        # print(outputs["instances"].pred_boxes)

        # im은 BRG이기 때문에 RGB로 convert == im[:, :, ::-1] 
        v = Visualizer(im[:, :, ::-1], metadata=dataset_metadata, scale=1.0)

        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        file_name = os.path.basename(path_)
        print(f"#            file_name : {file_name}")
        cv2.imwrite(os.path.join(result_dir,file_name), out.get_image()[:, :, ::-1] )


        if c_cfg.SAVE_DRAW_IMAGE : 
            json_dict = paprika_algorithm(find_condig, class_name_list, outputs, img, file_name, result_dir, args.type)

        # show result of inference
        #cv2.imshow("im", out.get_image()[:, :, ::-1])       # cv2_imshow(out.get_image()[:, :, ::-1])
        #while True:
        #    if cv2.waitKey() == 27:
        #        break
        end = time.time() 
        total_time += (end - start)
        print(f"#            {end - start:.5f} sec")

    print_time(total_time, num_images)
    


if __name__ == '__main__':
    c_cfg = InferenceConfig
    parser = argparse.ArgumentParser(
        description="decide what action to take"
    )
    parser.add_argument(
        "--dir",
        help="directory where images are stored for inference.", default= c_cfg.TEST_DIR ,
        type=str,
    )
    parser.add_argument(
        "--type",
        help="type of fruit", default= 'paprika' ,
        type=str,
    )
    parser.add_argument(
        "--model",
        help="trained model name", default= c_cfg.MODEL_NAME ,
        type=str,
    )
    args = parser.parse_args()

    if args.type is None:
        print("# Enter fruit type.") 
        sys.exit()
    elif args.type not in c_cfg.FRUIT_TYPE:
        print(f"# Enter fruit type among in {c_cfg.FRUIT_TYPE}")
        sys.exit()

    
    inference(c_cfg, args)