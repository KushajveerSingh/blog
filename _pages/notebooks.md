---
layout: page
title: Projects
permalink: /projects/
---

Check this [repo](https://github.com/KushajveerSingh/deep_learning) for more details.

* [Unsupervised Parking Lot Detection](https://github.com/KushajveerSingh/Unsupervised-Parking-Lot-Detection)
  
    A complete approach to detect parking lot spaces from images and then tell which spaces are occupied or not. Here I do not use any dataset for training my model to detect parking spaces. My implementation can be divided into these three modules:
    - *Object Detection Module* :- Use COCO pretrained model, no need to do finetuning.
    - *Label Processing Module* :- As out model is not finetuned, there are some tricks that I add to overcome these limitations
    - *Classification Module* :- Use the processed labels/bounding_boxes to tell if that parking space is occupied or not.
* [SPADE by Nvidia](https://github.com/KushajveerSingh/SPADE-PyTorch)
  
    Unofficial implementation of SPDAE for image-to-translation from segmentation maps to the colored pictures. Due to compute limit I test it out for a simplified model on Cityscapes dataset and get descent results after 80 epochs with batch_size=2.

* [Waste Seggregation using trashnet](https://github.com/KushajveerSingh/deep_learning/tree/master/projects/Waste_Seggregation_using_trashnet)

    Contains the code to train models for trashnet and then export them using ONNX. It was part of a bigger project where we ran these models on Rasberry Pi, which controlled wooden planks to classify the waste into different categories (code for rasberry pi not included here).

* [Unscramble game](https://github.com/KushajveerSingh/deep_learning/tree/master/random/unscramble_android_game)
  
    Python script to solve the unscramble android game. You are given 5 random letters and you have to find 3-letter, 4-letter, 5-letter english words from these 5 random letters. It is a simple brute force method with a english dictionary lookup.

* [Random Duty List](https://github.com/KushajveerSingh/Random-Duty-List)
  
    A PHP and MySQL based work where the aim is to assign duties from a list to various stations and make sure the duties are not repeated and the repetition occurs only after the list is exhausted.


## Jupyter Notebooks

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