# Clustering of Images using SCAN with Visual Explainations

## Contents
1. [Objective](#objective)
0. [Installation](#installation)
0. [Training](#training)
0. [Steps to execute](#tutorial)

## Objective
To perform clustering on the image dataset using SCAN with visual explainations


## Installation
The code runs with recent Pytorch versions, e.g. 2.0.1. 
The most important packages can be installed as:
```shell
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install matplotlib scipy scikit-learn   # For evaluation and confusion matrix visualization
conda install faiss-gpu                       # For efficient nearest neighbors search 
conda install pyyaml easydict                 # For using config files
conda install termcolor                       # For colored print statements
conda install -c conda-forge grad-cam         # For visual explainations
```
We refer to the `requirements.txt` file for an overview of the packages in the environment we used to produce our results.

## Training
We have trained the SCAN model over the standard CIFAR100-20 image training dataset and persisted that model into the repository at `repository_eccv\cifar-20\scan\model.pth.tar`
Similarly, we have refined the model using self-labeling and persisted that model at `repository_eccv\cifar-20\selflabel\model.pth.tar`

### Steps to execute
The configuration files can be found in the `configs/` directory. The clustering procedure consists of the following steps:
- __STEP 1__: Perform K nearest neighbours on each image to find its candidate images i.e. `find_candidates.py`
- __STEP 2__: Perform clustering on candidate images using our model i.e. `app.py`
- __STEP 3__: Display evaluation charts i.e. `eval_charts.py`

For example, run the following commands to perform our method on CIFAR20:
```shell
python find_candidates.py --config_env configs/env.yml --config_exp configs/pretext/simclr_cifar20.yml
python app.py --n 100 --query 500 --config_exp configs/scan/scan_cifar20.yml --model repository_eccv/cifar-20/scan/model.pth.tar
python eval_charts.py --config_exp configs/scan/scan_cifar20.yml --model repository_eccv/cifar-20/scan/model.pth.tar
```

