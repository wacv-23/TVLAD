# Weak-supervised Visual Geo-localization


## Introduction
`WAKD` is a PyTorch implementation for our ICPR-2022 paper "Weak-supervised Visual Geo-localization ……".


## Installation
We test this repo with Python 3.8, PyTorch 1.9.0, and CUDA 10.2. But it should be runnable with recent PyTorch versions (Pytorch >=1.0.0).
```shell
python setup.py develop
```


## Preparation
### Datasets

We test our models on three geo-localization benchmarks, [Pittsburgh](https://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Torii_Visual_Place_Recognition_2013_CVPR_paper.pdf), [Tokyo 24/7](https://www.di.ens.fr/~josef/publications/Torii15.pdf) and [Tokyo Time Machine](https://arxiv.org/abs/1511.07247) datasets. The three datasets can be downloaded at [here](https://www.di.ens.fr/willow/research/netvlad/).

The directory of datasets used is like
```shell
datasets/data
├── pitts
│   ├── raw
│   │   ├── pitts250k_test.mat
│   │   ├── pitts250k_train.mat
│   │   ├── pitts250k_val.mat
│   │   ├── pitts30k_test.mat
│   │   ├── pitts30k_train.mat
│   │   ├── pitts30k_val.mat
│   └── └── Pittsburgh
│           ├──images/
│           └──queries/
└── tokyo
    ├── raw
    │   ├── tokyo247
    │   │   ├──images/
    │   │   └──query/
    │   ├── tokyo247.mat
    │   ├── tokyoTM/images/
    │   ├── tokyoTM_train.mat
    └── └── tokyoTM_val.mat
```

### Pre-trained Weights

The file tree we used for storing the pre-trained weights is like
```shell
logs
├── vgg16_pretrained.pth.tar # refer to (1)
├── mbv3_large.pth.tar
└── vgg16_pitts_64_desc_cen.hdf5 # refer to (2)
└── mobilenetv3_large_pitts_64_desc_cen.hdf5
```

**(1) ImageNet-pretrained weights for CNNs backbone**

The ImageNet-pretrained weights for CNNs backbone or the pretrained weights for the whole model.

**(2) initial cluster centers for VLAD layer**

Note that the VLAD layer cannot work with random initialization.
The original cluster centers provided by NetVLAD or self-computed cluster centers by running the scripts/cluster.sh.

```shell
./scripts/cluster.sh mobilenetv3_large
```

## Training
Train by running script in the terminal. Script location: scripts/train_wakd_st.sh

Format:
```shell
bash scripts/train_wakd_st.sh arch archT
```
where, **arch** is the backbone name, such as mobilenetv3_large.
       **archT** is the teacher backbone name, such as vgg16.

For example:
```shell
bash scripts/train_wakd_st.sh mobilenetv3_large vgg16
```

In the train_wakd_st.sh.
In case you want to fasten testing, enlarge GPUS for more GPUs, or enlarge the --tuple-size for more tuples on one GPU.
In case your GPU does not have enough memory, reduce --pos-num or --neg-num for fewer positives or negatives in one tuple.

## Testing
Test by running script in the terminal. Script location: scripts/test.sh

Format:
```shell
bash scripts/test.sh resume arch dataset scale
```
where, **resume** is the trained model path.
       **arch** is the backbone name, such as vgg16, mobilenetv3_large and resnet152.
       **dataset scale**, such as pitts 30k and pitts 250k.

For example:
1. Test mobilenetv3_large on pitts 250k:
```shell
bash scripts/test.sh logs/netVLAD/pitts30k-mobilenetv3_large/model_best.pth.tar mobilenetv3_large pitts 250k
```
2. Test vgg16 on tokyo:
```shell
bash scripts/test.sh logs/netVLAD/pitts30k-vgg16/model_best.pth.tar model_best.pth.tar vgg16 tokyo
```
In the test.sh.
In case you want to fasten testing, enlarge GPUS for more GPUs, or enlarge the --test-batch-size for larger batch size on one GPU.
In case your GPU does not have enough memory, reduce --test-batch-size for smaller batch size on one GPU.

## Acknowledgements
We truely thanksful of the following two piror works. Particularly, part of the code is inspired by [[pytorch-NetVlad]](https://github.com/Nanne/pytorch-NetVlad)
+ NetVLAD: CNN architecture for weakly supervised place recognition (CVPR'16) [[paper]](https://arxiv.org/abs/1511.07247) [[pytorch-NetVlad]](https://github.com/Nanne/pytorch-NetVlad)
+ SARE: Stochastic Attraction-Repulsion Embedding for Large Scale Image Localization (ICCV'19) [[paper]](https://arxiv.org/abs/1808.08779) [[deepIBL]](https://github.com/Liumouliu/deepIBL)
