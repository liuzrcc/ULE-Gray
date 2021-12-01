This repository releases the code of our paper ["Going Grayscale: The Road to Understanding and Improving Unlearnable Examples"](https://arxiv.org/abs/2111.13244).

### Overview

We first show that grayscale pre-filtering (S1: Reactive Exploiter) can be used to mitigate original unlearnable examples ([Huang et al. 2021](https://openreview.net/forum?id=iAmZUo0DxC0)). 
Then, we propose adaptive unlearnable examples (S2: Adaptive Defender) that are resistant to grayscale pre-filtering. 
We also show that unlearnable examples generated using Multi-Layer Perceptrons (MLPs) are effective in the case of complex Convolutional Neural Network (CNN) classifiers.

<p align="center">
<img src="/figures/diagram_sce_2.PNG" width="500">
</p>

Figure above shows the clean test accuracies of ```ResNet-18``` on CIFAR-10 in S1 and S2 scenarios.

<br/><br/>


Our code is mainly based on and extended from the publicly available [Unlearnable Example implementation](https://github.com/HanxunH/Unlearnable-Examples/).

### How to use the code:

### 1. Generate ULEs in our S1 and S2 scenarios:

```
bash ./scripts/cifar10_poison/cifar10_ULEO/resnet18.sh
bash ./scripts/cifar10_poison/cifar10_ULEO_GRAYAUG/resnet18.sh
```


#### Train a standard exploiter on ULEOs:

```
bash ./scripts/cifar10_train/standard_exploiter/cifar10_ULEO/resnet18.sh
```

#### Train a standard exploiter on ULEO-GrayAugs:

```
bash ./scripts/cifar10_train/standard_exploiter/cifar10_ULEO_GRAYAUG/resnet18.sh
```

#### Train a grayscale exploiter on ULEOs:

```
bash ./scripts/cifar10_train/gray_exploiter/cifar10_ULEO/resnet18.sh
```

#### Train a grayscale exploiter on ULEO-GrayAugs:

```
bash ./scripts/cifar10_train/gray_exploiter/cifar10_ULEO_GRAYAUG/resnet18.sh
```


ULE perturbations can be found in ```./experiments/cifar10_poison/```.
After running the scripts, training results can be found in the folder ```./experiments/cifar10_train/```.


#### Generate ULEs on MLPs and test transferability:

### 2. Generate ULEOs on MLPs:

```
bash scripts/cifar10_poison/cifar10_ULEO/mlp.sh
```

#### Test transferability to three CNNs:

```
bash ./scripts/cifar10_train/standard_exploiter/cifar10_ULEO/mlp2dense.sh
bash ./scripts/cifar10_train/standard_exploiter/cifar10_ULEO/mlp2resnet.sh
bash ./scripts/cifar10_train/standard_exploiter/cifar10_ULEO/mlp2vgg.sh
```


After running the scripts, the transferability results can be found in the folder ```./experiments/cifar10_cross/mlp2x/```.

### More examples:

<p align="center">
<img src="/figures/moreuleo.png" width="300">

<img src="/figures/moreuleograyaug.png" width="304">
</p>

Unlearnable examples on CIFAR-10 by original approach (left) and ours (right). In each case, the order is the original image, unlearnable perturbations, and perturbed images.  

### Cite our work:

Please cite our paper and the original unlearnable example paper if you use this implementation in your research.

```
@misc{liu2021going,
      title={Going Grayscale: The Road to Understanding and Improving Unlearnable Examples}, 
      author={Zhuoran Liu and Zhengyu Zhao and Alex Kolmus and Tijn Berns and Twan van Laarhoven and Tom Heskes and Martha Larson},
      eprint={2111.13244},
      archivePrefix={arXiv},
      year={2021}
}
```
### Acknowledgement:

```
@inproceedings{huang2021unlearnable,
    title={Unlearnable Examples: Making Personal Data Unexploitable},
    author={Hanxun Huang and Xingjun Ma and Sarah Monazam Erfani and James Bailey and Yisen Wang},
    booktitle={ICLR},
    year={2021}
}
```
