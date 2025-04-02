# README 



#### directory map

> ◆ 표시 된 dir은 source code로 인해 생성되는 dir

```
ls 	┬ detectron2 (dir) 
	│
	├ models (dir) ◆
	│
	├ test_images (dir)
	│
	├ train_dataset (dir)  ┬   train (dir)  -  train_dataset.json
	│					   └   val_result_image (dir)	◆
	├ requirements.txt 	   
	│
	├ labeling (dir) - README.md
	│
	├ GT_segmentation_viewer.py
	│
	├ GT_image (dir) 	◆
	│
	├ train.py
	│
	├ inference.py
	│
	├ inference_result_image (dir)	◆
	│
	├ utils.py
	│
	├ config.py	
    │
    └ for_inference ┬ paprika.json
    				├ melon.json
```

- `detectron2 (dir)` : import `detectron2`을 위한 package

- `models (dir)` : `train.py`으로 인해 학습된 model이 저장되는 dir

  `inference.py` 에서 load weight를 하는 model의 경로

- `test_images (dir)` : `inference.py ` 에서 model의 inference에 사용되는 images를 저장한 dir

- `train_dataset (dir)` : `train.py` 에서 model의 training에 사용되는 images를 저장한 dir

  - `train (dir)` : training과정 중 train에 사용되는 images와, 해당 images의 metadata가 mapping된 `train_dataset.json`가 저장된 dir

    > `train_dataset.json` 에 original image를 함께 mapping하면 너무 용량이 커지는 문제가 발생하여, image를 같은 directory에 위치시킨 후 filename을 통해 path를 tracking하여 image를 load해 training에 사용한다.

  - `val_result_image (dir)` : `train.py` 에서 `val_dataset.json` 에 의해 진행된 validation 의 결과를 image로 저장하는 dir

    `train.py` 에 의해 생성된다.

- `requirements.txt` : 해당 repository를 실행하기 위한 module 및 package에 관한 정보(최초 실행 전 반드시 확인)

- `labeling (dir)` : image에 GT data를 그려 json. file로 저장할 수 있도록 해주는 program이 포함된 dir

  자세한 건 해당 dir안의 `README.md`를 읽어볼 것 

- `GT_segmentation_viewer.py` : `train_dataset.json` 을 통해 GT data를 original image위에 projection하여 `GT_image (dir)` 에 저장한다.

- `GT_image (dir)` 

- `train.py` : `train_dataset (dir)` 안의 data를 기반으로 training과 validation을 진행한다.

  validation 결과는 `val_result_image (dir)` 에  .jpg로 저장한다.

  trained model은 `models (dir)` 에 저장한다.

- `inference.py` : `models (dir)`에서 특정 model의 weight를 load한 후, `test_images (dir)` 의 images에 대해 inference를 진행한다.

  inference 결과는 `inference_result_image (dir)` 에 .jpg로 저장한다.

- `utils.py` : `train.py`, `inference.py`에서 사용되는 여러 utility function이 define되어 있다.

- `config.py` : `train.py`, `inference.py`에서 사용되는 여러 configuration이 class로 define되어 있다.

- `for_inference` : `train_dataset.json` 에 의해 학습 된 model의 inference시 visualization을 위한 data를 담은 json file들을 모아놓은 dir

  - `paprika.json` 



## Detectron2

### Install

1. create virtual environment

   ```
   $ conda create -n detectron2 python==3.7
   ```

   ```
   $ conda activate detectron2
   ```

2. download Cuda 10.1

3. install pytorch

   ```
   # conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -c conda-forge
   ```

   > check cudatoolkit by `nvcc -V`

4. pip install 1

   1. cython

      ```
      $ pip install cython
      ```

   2. pycoco

      - linux

        ```
        $ pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
        ```

        

      - windows

        ```
        $ pip install pycocotools-windows
        ```

        

   3. opencv

      ```
      $ pip install opencv-python
      ```

      

5. install detectron 2

   ```
   $ mkdir detectron2
   $ cd detectron2
   ```

   ```
   $ git clone https://github.com/facebookresearch/detectron2.git
   ```

   > 안되면
   >
   > ```
   > $ git clone https://github.com/augmentedstartups/detectron2
   > ```

6. install VC_redist.x64  (download from web site)

   execut in directory named detectoron2

7. pip install 2

   ```
   $ pip install -e .
   ```

   > in directory detectoron2

   ```
   pip install fvcore==0.1.1.post20200716
   ```

8. test

   ```
   python tests/test_windows_install.py
   ```

   > image하나 떠야함
   >
   > image주소 바꾸려면 
   >
   > `res = requests.get("https://live.staticflickr.com/700/33224654191_fdaee2e3f1_c_d.jpg")`



[documentation](https://detectron2.readthedocs.io/en/latest/index.html) 

---





## Getting start

```
$ conda activate detectron2
```

> ```
> C:\Users\ITC\Desktop\Noh_TaeUk\github\detectron2\detectron2
> ```
>
> 여기에서 실행



#### 1. GT_segmentation_viewer.py

show or save dataset image with segmentation 

```
$ python GT_segmentation_viewer.py
```



- **directory map**

  input dataset

  ```
   cwd  --  train_dataset	┌	train 	- train_dataset.json
                          │
                          │
                          └ 	val		- val_dataset.json
  ```

  > train directory에 있는 train_dataset.json으로 GT image를 보려면 해당 directory에 train_dataset.json에 저장된 정보의 출처인 image들도 존재해야 한다.





#### 2. train.py

```
$ python train.py --type melon --iter 12345
```



- **dataset map (json)**

  ```python
  json_file = os.path.join(img_dir, "dataset.json")
      with open(json_file) as f:
          imgs_anns = json.load(f)
  ```

  - using `labelme2coco`

    ```
    annotations[0] = {
    	'segmentation' = regions_of_segmentation(float)
    	'iscrowd' = 0 or 1 (crowd여부)
        'area' = area_instance(float)
        'image_id' = id_image(int) # instance가 속한 image의 id
        'bbox' =  [x_min, y_min, x_value, y_value]  # x_min + x_value = x_max (=y)
        'category_id' = 0, 1, ... (int)
        'id' = instance_id
    }
    
    annotations = {
    	[ 0, 
    	  1,
    	  2,
    	  ...
    	  ...
    	  N-1,
    	  N	# count_instance
    	]
    }
    
    
    # information_categories
    categories = {
    	[	{
                'supercategory' = class_category(str)
                'id' = class_id(int)
                'name': class_label(str)
    		}
        ],
    	[	{
                'supercategory' = class_category(str)
                'id' = class_id(int)
                'name': class_label(str)
    		}
        ],
        ... # count_class
    }
    
    # information_images
    images = { 
    	[
    		{
    		 'height' = image_height (int)
    		 'width' = image_width (int)
    		 'id' = image_id	(int)
    		 'file_name' = name_file 	(str)
    		},
    		{
    		
    		},
    		...
    		{
    		
    		}, # count_image
    	]
    
    }
    
    imgs_anns = {
    	'images'
        'categories'
        'annotations'
    }
    ```
    
    ```
    segmentation = [
    y_0, x_0, y_1, x_1, y_2, x_2, ...
    ```

  

  - input data map

    ```
    annotations[0] = {
    	'bbox' = [x_min, y_min, x_max, y_max]
    	'bbox_mode', 
    	'segmentation', 
    	'category_id'
    }
    
    annotations = [
    	0, 
    	1, 
    	...,
    	...
    	N-1,
    	N	# count_instance
    ]
    
    dataset_dicts[0] = {
    	'file_path' = 'dataset/train/image_0.jpg'
    	'image_id' =  image_id	(int)
    	'height' = image_height (int)
    	'width' = image_width (int)
    	'annotations'
    }
    
    
    dataset_dicts = [
    	0,
    	1,
    	...
    	...
    	N-1,
    	N, 	# count_image
    ]
    ```

    




#### 3. inference.py

do inference using pre-trained model

```
$ python batch_inference.py --type melon --model model_melon_begonly.pth
```







