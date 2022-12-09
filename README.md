# TransVLAD for Visual Geo-localization


## Introduction
`TVLAD` is a PyTorch implementation for our wacv-2023 paper "TransVLAD: Multi-Scale Attention-Based Global Descriptors for Visual Geo-Localization". This work was part of a project with SNCF. If you use this code for your research, please cite our paper. For additional questions contact us via yifanxu98@163.com or pshams55@gmail.com.


## Installation
We test this repo with Python 3.8, PyTorch 1.9.0, and CUDA 10.2. However, it should be runnable with recent PyTorch versions (Pytorch >=1.1.0).
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
├── mbv3_large.pth.tar # refer to (1)
└── mobilenetv3_large_pitts_64_desc_cen.hdf5 # refer to (2)
```

**(1) ImageNet-pretrained weights for CNNs backbone**

The ImageNet-pretrained weights for CNNs backbone or the pretrained weights for the model.

**(2) initial cluster centers for VLAD layer**

Note that the VLAD layer cannot work with random initialization.
The original cluster centers provided by NetVLAD or self-computed cluster centers by running the scripts/cluster.sh.

```shell
bash scripts/cluster.sh mobilenetv3_large
```

## Training
Train by running script in the terminal. Script location: scripts/train_tvlad_dist.sh

Format:
```shell
bash scripts/train_tvlad_dist.sh arch
```
where, **arch** is the backbone name, such as mobilenetv3_large.

For example:
```shell
bash scripts/train_tvlad_dist.sh mobilenetv3_large
```

In the train_tvlad_dist.sh.
In case you want to fasten testing, enlarge GPUS for more GPUs, or enlarge the --tuple-size for more tuples on one GPU.
In case your GPU does not have enough memory, reduce --pos-num or --neg-num for fewer positives or negatives in one tuple.

## Testing
Test by running script in the terminal. Script location: scripts/test_dist.sh

Format:
```shell
bash scripts/test_dist.sh resume arch dataset scale
```
where, **resume** is the trained model path.
       **arch** is the backbone name, such as mobilenetv3_large.
       **dataset scale**, such as pitts 30k, pitts 250k, tokyo.

For example:
1. Test mobilenetv3_large on pitts 250k:
```shell
bash scripts/test_dist.sh logs/netVLAD/pitts30k-mobilenetv3_large/model_best.pth.tar mobilenetv3_large pitts 250k
```
In the test.sh.
In case you want to fasten testing, enlarge GPUS for more GPUs, or enlarge the --test-batch-size on one GPU.
In case your GPU does not have enough memory, reduce --test-batch-size on one GPU.
