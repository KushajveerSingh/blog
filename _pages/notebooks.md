---
layout: page
title: Notebooks
permalink: /notebooks/
---

I use [Pytorch](https://pytorch.org/) and [fastai](https://www.fast.ai/) as my main deep learning libraries. Check this [repo](https://github.com/KushajveerSingh/deep_learning) for more details.

Summary of few of my jupyter notebooks

* **Mish activation function** is tested for transfer learning. Here mish is used only in the last fully-connected layers of a pretrainened Resnet50 model. I test the activation function of CIFAR10, CIFAR100 using three different learning rate values. I found that Mish gave better results than ReLU. [notebook](https://github.com/KushajveerSingh/deep_learning/tree/master/paper_implementations/Study%20of%20Mish%20activation%20function%20in%20transfer%20learning%20with%20code%20and%20discussion), [paper](https://arxiv.org/abs/1908.08681)

* **Multi Sample Dropout** is implemented and tested on CIFAR-100 using cyclic learning. My losses converged 4x faster when using num_samples=8 than using simple dropout. [notebook](https://github.com/KushajveerSingh/deep_learning/tree/master/paper_implementations/Multi%20Sample%20Dropout), [paper](https://arxiv.org/abs/1908.08681)

* **Data Augmentation in Computer Vision**
    - Notebook implementing single image data augmentation techniques using just Python [notebook](https://github.com/KushajveerSingh/deep_learning/tree/master/paper_implementations/Data%20Augmentation%20in%20Computer%20Vision)

* **Summarizing Leslie N. Smith’s research** in cyclic learning and hyper-parameter setting techniques. [notebook](https://github.com/KushajveerSingh/deep_learning/tree/master/paper_implementations/Leslie%20N.%20Smith%20papers%20notebook)
    - A disciplined approach to neural network hyper-parameters: Part 1 – learning rate, batch size, momentum, and weight decay [paper](https://arxiv.org/abs/1803.09820)
    - Super-Convergence: Very Fast Training of Neural Networks Using Learning Rates [paper](https://arxiv.org/abs/1708.07120)
    - Exploring loss function topology with cyclical learning rates [paper](https://arxiv.org/abs/1702.04283)
    - Cyclical Learning Rates for Training Neural Networks [paper](https://arxiv.org/abs/1506.01186)

* **Photorealisitc Style Transfer**. Implementation of the *High-Resolution Network for Photorealistic Style Transfer paper*. [notebook](https://github.com/KushajveerSingh/Photorealistic-Style-Transfer), [paper](https://arxiv.org/abs/1904.11617)

* **Weight Standardization** is implemented and tested using cyclic learning. I find that it does not work well with cyclic learning when using CIFAR-10. [notebook](https://github.com/KushajveerSingh/deep_learning/tree/master/paper_implementations/weight_standardization), [paper](https://arxiv.org/abs/1903.10520)

* **Learning Rate Finder**. Implementation of learning rate finder as introduced in the paper [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1903.10520). A general template for custom models is provided. [notebook](https://github.com/KushajveerSingh/fastai_without_fastai/blob/master/notebooks/lr_find.ipynb)

* **PyTorch computer vision tutorial**. AlexNet with tips and checks on how to train CNNs. The following things are included: [notebook](https://github.com/KushajveerSingh/deep_learning/tree/master/paper_implementations/Training%20AlexNet%20with%20tips%20and%20checks%20on%20how%20to%20train%20CNNs)
    - Dataloader creation
    - Plotting dataloader results
    - Weight Initialization
    - Simple training loop
    - Overfitting a mini-batch

* [How to deal with outliers](https://github.com/KushajveerSingh/deep_learning/tree/master/paper_implementations/How%20to%20deal%20with%20outliers)

* [How to choose number of bins in a histogram](https://github.com/KushajveerSingh/deep_learning/tree/master/paper_implementations/Number%20of%20bins%20of%20a%20Histogram)