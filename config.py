


class TrainConfig():
    # name of custom dataset
    DATASET_NAME = 'hibernation'
    SCORE_THRESHOLD = 0.5
    # COMPUTE_ITER = True 가장 최적의 iter을 계산. False이면 아래 설정한 ITER만큼 학습
    COMPUTE_ITER = False
    ITER = 100000
    LEARNING_LEATE = 0.00005

    EPOCH = 1000
    # EVAL_PERIOD = 200
    BATCH_SIZE = 4

    # 기존에 있던 model을 load해서 이어서 학습
    # False이면initial training
    CONTINUE_LEARNING = False
    CONTINUE_MODEL_NAME = 'model_0114999.pth'

    

    # train dataset안에서 train에 쓰일 dataset의 비율(나머지 비율은 validation dataset으로 사용된다.)
    TRAIN_VAL_RATIO = 1.0
    TRAIN_EVALUATION_RATIO = None

    # directory of saved models
    MODELS_DIR = "models"

    # path of directory whitch located training, validation dataset
    TRAIN_DIR = 'train_dataset'
    # 'train_dataset'

    # dataset file for training 
    TRAIN_DATASET = 'train_dataset.json'

    # directory path of result of validation
    PATH_VAL_RESULT = TRAIN_DIR + '/val_result_image'

    VAL_BATCH_SIZE = 4


class GT_imageConfig(TrainConfig):
    GT_DIR_NAME = 'GT_image'

    ### options
    
    # True이면 GT image만 따로 저장
    # False이면 각 GT image에 대한 pay를 계산하여 excel file에 저장
    JUST_VIEW_GTIMAGE = False
    EXCEL_FILE_NAME = "hibernation.xlsx"

    # True이면 excel file에 GT image삽입
    INSERT_IMAGE_EXCEL = False


class InferenceConfig(TrainConfig):

    # path of directory whitch located test images
    TEST_DIR = 'test_images'
    # test_images

    # directory path of result of inference
    PATH_INFER_RESULT = 'inference_result_image'

    LOAD_MODEL_DIR = "models"

    SCORE_THRESHOLD = 0.5

    # name of model file for load weight
    MODEL_NAME = "model_paprika_38600.pth"
    # "model_midrid.pth"
  
    FRUIT_TYPE = ['paprika', 'melon', "cucumber", "onion", "strawberry", 
                  'tomato',
                  "tomato_seed", "chilipepper_seed", "cucumber_seed",
                  "bug", "pear",
                  "seedling_board", "watermelon_seed", "strawberry_disease"
                  ]

    ### options

    RESIZE_IMAGE = False

    # True : detection된 object정보 출력
    PRINT_OBJECT_INFO = False

    # True : find algorithm 적용 
    SAVE_RESULT = True

    # show found coordinates
    DRAW_COORDINATES = True

    INTER_BATCH_SIZE = TrainConfig.VAL_BATCH_SIZE

    JSON_PATH_FOR_INFERENCE = "for_inference"




class EvaluationConfig(TrainConfig):

    # True : save ground truth, inference result image 
    SAVE_FLAG = True

    TRAIN_DIR = 'train_dataset'

    TRAIN_DATASET = 'train_dataset.json'
    
    THRESHOLD = [0.5, 0.9]

    MODELS_DIR = TrainConfig.MODELS_DIR

    EVALUATION_BATCH_SIZE = TrainConfig.VAL_BATCH_SIZE

    SCORE_THRESHOLD = TrainConfig.SCORE_THRESHOLD

    PATH_EVAL_RESULT = 'evaluation_result'

    TRAIN_EVALUATION_RATIO = 0.9
    TRAIN_VAL_RATIO = None

    JSON_PATH_FOR_EVALUATION = "for_inference"




class Find_algorithm_config():
    # midrid를 잇는 점의 개수 (polygon기준)
    # 제대로 알고리즘이 적용된다면 6을 입력했을 시 전체 midrid를 잇는 점은 10개가 표시된다.
    # 값이 클 수록 좌표의 정확도 증가
    if type == "strawberry" : 
        NUM_SPOT = 10
    else:
        NUM_SPOT = 20

    # boundary coordinates of leaf중 midrid points 두 개로 그려지는 1차함수 위의 점을 찾을 때의 오차범위 
    # 0이 가장 이상적이다. (model의 성능이 가장 좋은 경우)
    # midrid의 first 또는 last point가 boundary of leaf까지 이어지지 않는다면 해당 값을 올려서 오차범위를 넓혀 탐색하도록 하자.
    MARGIN_ERROR = 5

    # fruit에서 cap이 fruit의 중앙의 얼마만큼의 범위 내에 속하는지를 결정하는 값
    FRUIT_CENTER_BOX_RATIO = 0.2

    RESIZE_SCALE = 1

        

    
    
       

