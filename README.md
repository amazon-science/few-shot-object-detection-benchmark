# Rethinking Few-Shot Object Detection on a Multi-Domain Benchmark
This is the PyTorch implementation of Our ECCV paper.  It proposes a benchmark for multi-domain few-shot object detection.


![Datasets](figs/datasets_0.png)



If you use the code/model/results of this repository please cite:

    @inproceedings{lee2022mofsod,
      author    = {Kibok Lee and
                   Hao Yang and
                   Satyaki Chakraborty and
                   Zhaowei Cai and
                   Gurumurthy Swaminathan and
                   Avinash Ravichandran and
                   Onkar Dabeer},
      title     = {Rethinking Few-Shot Object Detection on a Multi-Domain Benchmark},
      booktitle = {ECCV}},
      year      = {2022},
    }

Please also cite all the datasets used in the paper. 
If you want to run the baseline experiments, please cite the corresponding paper as well.

## Installation

In this repo, we mainly provide the scripts needed to run MoFSOD benchmarks on top of several few-shot/general detection methods.
To run baselines in the paper, you need to checkout these repos:
[DeFRCN](https://github.com/er-muyue/DeFRCN), 
[FSCE](https://github.com/megvii-research/FSCE), 
[FsDet](https://github.com/ucbdrive/few-shot-object-detection)
,[UniDet](https://github.com/xingyizhou/UniDet) and [Detic](https://github.com/facebookresearch/Detic). 
Please follow each repo's instructions to install. 

All of the above codebasese are based on [Detectron2](https://github.com/facebookresearch/detectron2).
Due to compatibility issues, we use [Detectron2-v0.3](https://github.com/facebookresearch/detectron2/tree/v0.3) for FSCE and FsDet,
and [Detectron2-v0.5](https://github.com/facebookresearch/detectron2/tree/v0.5) for DeFRCN, UniDet and Detic. Our suggestions is to build two docker images, one with [Detectron2-v0.3](https://github.com/facebookresearch/detectron2/tree/v0.3) and the other with
[Detectron2-v0.5](https://github.com/facebookresearch/detectron2/tree/v0.5) and use separate dockers for different experiments. 


## Dataset Preparation

Please refer the [readme](/datasets/README.md) under `datasets` folder to setup all 10 datasets in the benchmark.  

We also provide the [converted OpenImages style annotations](/prosessed_anntoations/datasets.tar.gz) for easier setup. 
You still need to follow the [readme](/datasets/README.md) to download and setup JPEGImages folder, then extract and write the `datasets` folder with the 
[datasets]((/prosessed_anntoations/datasets.tar.gz)) tar file.

## Few-shot Sampling

To create the sampling we used in the paper, we prepare a script with fixed random seeds. After setting up the `datasets` folder, just run 

    python generate_oi_annotations.py

to create the few-shot sampled datasets. We also provide the [exact sampling](/prosessed_anntoations/few_shot_sampling.tar.gz) we used in the paper.


## Running on each baseline

Due to the testing needed, we will release this part in a few weeks, stay tune!
In the mean time, if you want to test the datasets, we add an OpenImages dataset [reader](/common/openimages_dataset.py) to use in Detectron2.
You can register the datasets with 

    from common.openimages_dataset import openimages
    dataset = openimages(data_dir='./datasets/', dataset_name=deepfruits, post_fix_key=str(fold))

