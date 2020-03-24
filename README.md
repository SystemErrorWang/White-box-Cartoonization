<img src='paper/shinjuku.jpg' align="left" width=1000>

<br><br><br>

# Learning to Cartoonize Using White-box Cartoon Representations
[project page](https://systemerrorwang.github.io/White-box-Cartoonization/) |   [paper](https://github.com/SystemErrorWang/White-box-Cartoonization/blob/master/paper/06791.pdf)

Tensorflow implementation for CVPR2020 paper “Learning to Cartoonize Using White-box Cartoon Representations”

<img src="images/method.jpg" width="1000px"/>
<img src="images/use_cases.jpg" width="1000px"/>

## Use cases
### People
<img src="images/person1.jpg" width="1000px"/>
<img src="images/person2.jpg" width="1000px"/>

### Scenery
<img src="images/city1.jpg" width="1000px"/>
<img src="images/city1.jpg" width="1000px"/>

### Animals ans Plants
<img src="imgs/animal.jpg" width="1000px"/>
<img src="imgs/plants.jpg" width="1000px"/>

### Food
<img src="images/food.jpg" width="1000px"/>

### Indoor Scenes
<img src="images/home.jpg" width="1000px"/>

### More Images Are Shown In The Supplementary Materials


## Prerequisites
- Training code: Linux or Windows
- NVIDIA GPU + CUDA CuDNN for performance
- Inference code: Linux, Windows and MacOS


## Getting Started
### Installation
- Assume you already have NVIDIA GPU and CUDA CuDNN installed 
- Install tensorflow-gpu, we tested 1.12.0 and 1.13.0rc0 
- Install scikit-image==0.14.5, other version may cause problems


### Inference with Pre-trained Model
- Store test images in /test_code/test_images
- Run /test_code/cartoonize.py
- Results will be saved in /test_code/cartoonized_images


### Train


## Datasets



## Citation
If you use this code for your research, please cite our [paper](https://systemerrorwang.github.io/White-box-Cartoonization/):

