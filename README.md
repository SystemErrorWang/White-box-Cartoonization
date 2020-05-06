<img src='paper/shinjuku.jpg' align="left" width=1000>

<br><br><br>

# [CVPR2020]Learning to Cartoonize Using White-box Cartoon Representations
[project page](https://systemerrorwang.github.io/White-box-Cartoonization/) |   [paper](https://github.com/SystemErrorWang/White-box-Cartoonization/blob/master/paper/06791.pdf) |   [twitter](https://twitter.com/IlIIlIIIllIllII/status/1243108510423896065) |   [zhihu](https://zhuanlan.zhihu.com/p/117422157) |   [bilibili](https://www.bilibili.com/video/av56708333)

Tensorflow implementation for CVPR2020 paper “Learning to Cartoonize Using White-box Cartoon Representations”.

This repo in under construction, now inference code is available, training code will be updated soon

<img src="images/method.jpg" width="1000px"/>
<img src="images/use_cases.jpg" width="1000px"/>

## Use cases

### Scenery
<img src="images/city1.jpg" width="1000px"/>
<img src="images/city2.jpg" width="1000px"/>

### Food
<img src="images/food.jpg" width="1000px"/>

### Indoor Scenes
<img src="images/home.jpg" width="1000px"/>

### People
<img src="images/person1.jpg" width="1000px"/>
<img src="images/person2.jpg" width="1000px"/>

### More Images Are Shown In The Supplementary Materials

## Prerequisites

- Training code: Linux or Windows
- NVIDIA GPU + CUDA CuDNN for performance
- Inference code: Linux, Windows and MacOS


## How To Use

### Installation

- Assume you already have NVIDIA GPU and CUDA CuDNN installed 
- Install tensorflow-gpu, we tested 1.12.0 and 1.13.0rc0 
- Install scikit-image==0.14.5, other versions may cause problems


#### macOS

```bash
brew install python3
pip3 install opencv-python tensorflow==1.15 tqdm
```


### Inference with Pre-trained Model

- Store test images in /test_code/test_images
- Run /test_code/cartoonize.py
- Results will be saved in /test_code/cartoonized_images


### Train

- Place your training data in corresponding folders in /dataset 
- Run pretrain.py, results will be saved in /pretrain folder
- Run train.py, results will be saved in /train_cartoon folder
- Codes are cleaned from production environment and untested
- There may be minor problems but should be easy to resolve


### Datasets

- Due to copyright issues, we cannot provide cartoon images used for training
- However, these training datasets are easy to prepare
- Scenery images are collected from Shinkai Makoto, Miyazaki Hayao and Hosoda Mamoru films
- Clip films into frames and random crop and resize to 256x256
- Portrait images are from Kyoto animations and PA Works
- We use this repo(https://github.com/nagadomi/lbpcascade_animeface) to detect facial areas
- Manual data cleaning will greatly increace both datasets quality


### Troubleshooting


#### ValueError: Can't load save_path when it is None

If you see the following error when running `python[3] ./test_code/cartoonize.py`, change the
current working directory to `test_code` and run the script from there.

```
Traceback (most recent call last):
  File "./test_code/cartoonize.py", line 65, in <module>
    cartoonize(load_folder, save_folder, model_path)
  File "./test_code/cartoonize.py", line 39, in cartoonize
    saver.restore(sess, tf.train.latest_checkpoint(model_path))
  File "/usr/local/lib/python3.7/site-packages/tensorflow_core/python/training/saver.py", line 1277, in restore
    raise ValueError("Can't load save_path when it is None.")
ValueError: Can't load save_path when it is None.
```


#### TensorFlow binary was not compiled to use: AVX2 FMA

If the installed `tf` fails with the following error, try following [these instructions](https://github.com/lakshayg/tensorflow-build).

```
Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
```


## Acknowledgement

We are grateful for the help from Lvmin Zhang and Style2Paints Research

## Citation

If you use this code for your research, please cite our [paper](https://systemerrorwang.github.io/White-box-Cartoonization/):

    Bib file coming soon.

# 中文社区

我们有一个除了技术什么东西都聊的以技术交流为主的宇宙超一流二次元相关技术交流吹水群“纸片协会”。如果你一次加群失败，可以多次尝试。

    纸片协会总舵：184467946
