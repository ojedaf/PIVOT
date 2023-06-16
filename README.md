# PIVOT: Prompting for Video Continual Learning.

```diff
- The PIVOT Code will be available SOON.
```

[[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Villa_PIVOT_Prompting_for_Video_Continual_Learning_CVPR_2023_paper.pdf)

Modern machine learning pipelines are limited due to data availability, storage quotas, privacy regulations, and expensive annotation processes. These constraints make it difficult or impossible to train and update large-scale models on such dynamic annotated sets. Continual learning directly approaches this problem, with the ultimate goal of devising methods where a deep neural network effectively learns relevant patterns for new (unseen) classes, without significantly altering its performance on previously learned ones. In this paper, we address the problem of continual learning for video data. We introduce PIVOT, a novel method that leverages extensive knowledge in pre-trained models from the image domain, thereby reducing the number of trainable parameters and the associated forgetting. Unlike previous methods, ours is the first approach that effectively uses prompting mechanisms for continual learning without any in-domain pre-training. Our experiments show that PIVOT improves state-of-the-art methods by a significant 27% on the 20-task ActivityNet setup.

![PIVOT-model](https://github.com/ojedaf/PIVOT/blob/main/images/img_model.png)

## Prerequisites

It is essential to install all the dependencies and libraries needed to run the project. To this end, you need to run this line: 

```
conda env create -f environment.yml
```
### Dataset

We leverage The vCLIMB Benchmark to evaluate PIVOT. For more information about the benchmark and how to set it, we encourage you to visit [The vCLIMB website](https://github.com/ojedaf/vCLIMB_Benchmark)

## Citation

If you find this repository useful for your research, please consider citing our paper:

```
@inproceedings{PIVOT_villa,
  author    = {Villa, Andr{\'{e}}s and
               Le{\'{o}}n Alc{\'{a}}zar, Juan and
               Alfarra, Motasem and
               Alhamoud, Kumail and
               Hurtado, Julio and
               Caba Heilbron, Fabian and
               Soto, Alvaro and
               Ghanem, Bernard},
  title     = {{PIVOT:} Prompting for Video Continual Learning},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023},
  month={June}
}
```
