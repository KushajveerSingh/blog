---
layout: page
title: Projects
permalink: /projects/
---

I use [Pytorch](https://pytorch.org/) and [fastai](https://www.fast.ai/) as my main deep learning libraries. Check this [repo](https://github.com/KushajveerSingh/deep_learning) for more details.

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
  
    A PHP and MySQL based work where the aim is to assign duties from a list to various stations and make sure the duties are not repeated and the repetition occurs only after the list is exhasuted.