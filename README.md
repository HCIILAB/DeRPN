# DeRPN:  Taking a further step toward more general object detection
DeRPN is a novel region proposal network which concentrates on improving the adaptivity of current detectors. 
The paper is available [here](https://arxiv.org/abs/1811.06700).

<div align="center"><img src="https://github.com/HCIILAB/DeRPN/blob/master/schema.png" width="600" ></div>

## Recent Update
**·** Mar. 13, 2019: The DeRPN pretrained models are added.

**·** Jan. 25, 2019: The code is released.

## Contact Us
Welcome to improve DeRPN together. For any questions, please feel free to contact Lele Xie (xie.lele@mail.scut.edu.cn) or Prof. Jin (eelwjin@scut.edu.cn).

## Citation
If you find DeRPN useful to your research, please consider citing our paper as follow:
```
@article{xie2019DeRPN,
  title     = {DeRPN: Taking a further step toward more general object detection},
  author    = {Lele Xie, Yuliang Liu, Lianwen Jin*, Zecheng Xie}
  joural    = {AAAI}
  year      = {2019}
}
```
## Main Results
**Note**: The reimplemented results are slightly different from those presented  in the paper for different training settings, but the conclusions are still consistent. For example, this code doesn't use multi-scale training which should boost the results for both DeRPN and RPN.

### COCO-Text

training data:  COCO-Text train

test data: COCO-Text test

|                   | network | AP@0.5  | recall@0.5 | AP@0.75 | recall@0.75 |
|:-----------------:|:---------------------:|:---------------------:|:---------------------:|:---------------------:|:---------------------:|
| RPN+Faster R-CNN | VGG16 | 32.48 | 52.54 | 7.40 | 17.59 |
| DeRPN+Faster R-CNN | VGG16 | 47.39 | 70.46 | 11.05 | 25.12 |
|RPN+R-FCN  |ResNet-101  | 37.71 | 54.35 | 13.17 | 22.21 |
|DeRPN+R-FCN |ResNet-101 | 48.62 | 71.30 | 13.37 | 27.57 |

### Pascal VOC

training data:  VOC 07+12 trainval 

test data: VOC 07 test

Inference time is evaluated on one TITAN XP GPU.

|                     |  network   | inference time | AP@0.5 | AP@0.75 |  AP   |
| :-----------------: | :--------: | :------------: | :----: | :-----: | :---: |
|  RPN+Faster R-CNN   |   VGG16    |     64 ms      | 75.53  |  42.08  | 42.60 |
| DeRPN+Faster R-CNN  |   VGG16    |     65 ms      | 76.17  |  44.97  | 43.84 |
|      RPN+R-FCN      | ResNet-101 |     85 ms      | 78.87  |  54.30  | 50.04 |
| DeRPN+R-FCN (900) * | ResNet-101 |     84 ms      | 79.21  |  54.43  | 50.28 |

( "*": On Pascal VOC dataset, we found that it is more suitable to train the DeRPN+R-FCN model with 900 proposals. For other experiments, we use the default proposal number to train the models, i.e., 2000 proposals fro Faster R-CNN, 300 proposals for R-FCN. )

### MS COCO

training data:  COCO 2017 train

test data: COCO 2017 test/val

| test set | network |  AP  | AP50 | AP75 | AP<sub>S</sub>  | AP<sub>M</sub>  | AP<sub>L</sub>  |
| :----------------: | :-----: | :--: | :----: | :-----: | ---- | ---- | :--: |
|  RPN+Faster R-CNN  |  VGG16  | 24.2 | 45.4 | 23.7 | 7.6 | 26.6 | 37.3 |
| DeRPN+Faster R-CNN |  VGG16  | 25.5 | 47.2 | 25.2 | 10.3 | 27.9 | 36.7 |
|     RPN+R-FCN      | ResNet-101 | 27.7 | 47.9 | 29.0 | 10.1 | 30.2 | 40.1 |
|    DeRPN+R-FCN     | ResNet-101 | 28.4 | 49.0 | 29.5 | 11.1 | 31.7 | 40.5 |


| val set | network |  AP  | AP50 | AP75 |AP<sub>S</sub>  | AP<sub>M</sub>  | AP<sub>L</sub>  |
| :----------------: | :-----: | :--: | :----: | :-----: | ---- | ---- | :--: |
|  RPN+Faster R-CNN  |  VGG16  | 24.1 | 45.0 | 23.8 | 7.6            | 27.8           |      37.8      |
| DeRPN+Faster R-CNN |  VGG16  | 25.5 | 47.3 | 25.0 | 9.9 | 28.8 | 37.8 |
|     RPN+R-FCN      | ResNet-101 | 27.8 | 48.1 | 28.8 | 10.4 | 31.2 | 42.5 |
|    DeRPN+R-FCN     | ResNet-101 | 28.4 | 48.5 | 29.5 | 11.5 | 32.9 | 42.0 |

## Getting Started

1. Requirements
2. Installation
3. Preparation for Training & Testing
4. Usage

## Requirements
1. Cuda 8.0 and cudnn 5.1. 
2. Some python packages: cython, opencv-python, easydict et. al. Simply install them if your system misses these packages.
3. Configure the caffe according to your environment ([Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html)). As the code requires pycaffe, caffe should be built with python layers. In Makefile.config, make sure to uncomment this line:

  ```
  WITH_PYTHON_LAYER := 1
  ```
4. An NVIDIA GPU with more than 6GB is required for ResNet-101.

## Installation
1. Clone the DeRPN repository
    ```
    git clone https://github.com/HCIILAB/DeRPN.git
    ```
2. Build the Cython modules
    ```Shell
    cd $DeRPN_ROOT/lib
    make
    ```

3. Build caffe and pycaffe
    ```Shell
    cd $DeRPN_ROOT/caffe
    make -j8 && make pycaffe
    ```

## Preparation for Training & Testing
### Dataset

1. Download the datasets of [Pascal VOC 2007 & 2012](http://host.robots.ox.ac.uk/pascal/VOC/), [MS COCO 2017](http://cocodataset.org/#download) and [COCO-Text](http://rrc.cvc.uab.es/?ch=5&com=introduction).

2. You need to put these datasets under the $DeRPN_ROOT/data folder (with symlinks). 

3. For COCO-Text, the folder structure is as follow:

    ```Shell
    $DeRPN_ROOT/data/coco_text/images/train2014
    $DeRPN_ROOT/data/coco_text/images/val2014
    $DeRPN_ROOT/data/coco_text/annotations  
    # train2014, val2014, and annotations are symlinks from /pth_to_coco2014/train2014, 
    # /pth_to_coco2014/val2014 and /pth_to_coco2014/annotations2014/, respectively.
    ```
4. For COCO, the folder structure is as follow:

    ```Shell
    $DeRPN_ROOT/data/coco/images/train2017
    $DeRPN_ROOT/data/coco/images/val2017
    $DeRPN_ROOT/data/coco/images/test-dev2017
    $DeRPN_ROOT/data/coco/annotations  
    # the symlinks are similar to COCO-Text
    ```

5. For Pascal VOC, the folder structure is as follow:

    ```Shell
    $DeRPN_ROOT/data/VOCdevkit2007
    $DeRPN_ROOT/data/VOCdevkit2012
    #VOCdevkit2007 and VOCdevkit2012 are symlinks from $VOCdevkit whcich contains VOC2007 and VOC2012.
    ```

### Pretrained models

Please download the ImageNet pretrained models ([VGG16](https://pan.baidu.com/s/1BDDl5xtrBznlyIrVj9g_zQ) and [ResNet-101](https://pan.baidu.com/s/1BDDl5xtrBznlyIrVj9g_zQ), password: k4z1), and put them under
```Shell
$DeRPN_ROOT/data/imagenet_models
```
We also provide the DeRPN pretrained models [here](https://pan.baidu.com/s/141Dy0OiXLMau-XLdEkHdIA) (password: fsd8).


## Usage
```Shell
cd $DeRPN_ROOT
./experiments/scripts/faster_rcnn_derpn_end2end.sh [GPU_ID] [NET] [DATASET]

# e.g., ./experiments/scripts/faster_rcnn_derpn_end2end.sh 0 VGG16 coco_text
```

## Copyright
This code is free to the academic community for research purpose only. For commercial purpose usage, please contact Dr. Lianwen Jin: lianwen.jin@gmail.com.
