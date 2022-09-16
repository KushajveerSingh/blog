---
layout: page
title: Projects
permalink: /projects/
---

### Table of contents
- [Work Experience](#work-experience)
- [Additional Experience](#additional-experience)
- [Projects](#projects)
- [Blogs](#blogs)
- [Certifications](#certifications)

### Work Experience
- **Full Stack Developer** (Jul 2022 - current, UGA)
    - Worked in the IT department of the College of Agriculture and Environmental Sciences, University of Georgia (Athens, GA, USA).
    - Migrated a legacy *Microsoft Access* database to *PostgreSQL* with 18 years of data in it.
    - In order to maintain the existing 217 legacy programs, wrote a parser in *Node.js* and *Express* which converts Access queries to PostgreSQL queries on the fly.
    - Created the development and production servers, which included doing IP whitelisting, backup configuration with a local storage server, periodically syncing development and production servers, and implementing the best Linux security practices.
    - To prevent people from accessing the SQL queries from outside, wrote a program in *Node.js* using which queries can be written in a text file, that is never shared with the public, and refer to them using a special function in *JavaScript*.
    - Automated the data entry pipeline for handwritten submission forms using *Python* and *JavaScript*, resulting in a reduction of time from 2 minutes per form to 25 seconds on average. This was done by adding autocomplete functionality for various fields of the form like name, address, contact info and merging that with the existing database interface. 
    - Used *Selenium* to automate browser-related tasks, like downloading an invoice from UPS and emailing it internally.
    - To send results to the client, a codebase built on top of a proprietary language created by the IT manager was being used which converts information from the database into a PDF. Rewrote that entire codebase using *React*, *TypeScript*, *Material-UI*, *Redux*. This included converting information from a 200-page manual to a React program and updating the queries from Access to PostgreSQL.
    - Agile development was used. Email and github issues were used to track and communicate daily status and a weekly retrospective meeting was conducted to review the successes of the week and areas of improvement from the previous week.

- **Graduate Teaching Assistant** (Jan 2022 - Aug 2022, UGA)
    - Worked in the Computer Science department, University of Georgia (Athens, GA, USA).
    - Led weekly meetings and facilitated group discussions for the course *Discrete Mathematics for CS*.
    - Held one-on-one office hours for 2 transfer students and went through the entire course content in detail, along with discussing practical applications of the course content.


### Additional Experience
- **Poster submission, School of Computing Research Day** (Sep 2022 - Oct 2022)
    - Submitted 2 posters for the School of Computing, Research Day at University of Georgia (Athens, GA, USA)
    - Created a GlobalConsistency module that improves the robustness of graph neural networks against adversarial attacks
    - Proposed module beats state-of-the-art in 3 out of 4 datasets, while adding 4 lines of code
    - Presented a poster on generating full human body 3D scans from random noise

- **Workshop paper reviewer, CVPR 2021** (Apr 2021)
    - Reviewed 2 papers for the *Biometrics Workshop, jointly with the Workshop on Analysis and Modeling of Faces and Gestures, CVPR 2021*
    - Split the review into summary, strengths, weaknesses

- **Paper reviewer and presentation judge, GJSHS 2021** (Feb 2021 - Mar 2021)
    - Reviewed 6 papers for the 46th *Georgia Junior Science & Humanities Symposium (GJSHS)* from high-school students.
    - The review included detailed feedback on the strengths and weaknesses of the paper and suggestions for improvement based on the recent research in the field
    - Presentation judge for the final round of the competition.

- **Winner TechGig CodeGladiators hackathon** (May 2019 - July 2019)
    - Won the first sole place in TechGig CodeGladiators, in the Artificial Intelligence theme where 15 teams were selected from all across India from a pool of 0.5 million candidates.
    - Implemented a fully-deployable parking space detection system using *PyTorch* from video.
    - Detections from multiple frames can be combined and adjusted over time depending on where the cars are being parked.
    - Frame can be split into a 3x3 grid allowing to make finer and more robust predictions and the result is combined into a single frame.  

- **Third place at IndiaSkills Nationals** (Sep 2018 - Nov 2018)
    - Won third place at the North-Zone regional finals in *IT Software Solutions for Business* at IndiaSkills, representing my state Chandigarh (India)
    - Build a desktop application using *C#* and an android application using *Java*.
    - Both applications required a database setup using *MySQL* and *phpMyAdmin*.

- **8-bit computer at PEC IEEE showcase** (Feb 2017 - Apr 2017)
    - Created an 8-bit computer on breadboard using NAND chips.
    - Implemented Add, Subtract logic and 2 byte memory module using *Flip Flops*
    - Showcased the work to high school students at Punjab Engineering College (Chandigarh, India) IEEE Project showcase

### Projects
- **JSON Graph Visualizer** (Sep 2022, UGA \| Athens, GA)
    - Build *Next.js*/*TypeScript*/*Redux*/*Material-UI*/*styled-components* application to convert JSON data into a graph
    - Generated multi-level graph allows easier reading of JSON data
    - Deployed the application on Github Pages

- **Sorting Visualizer** (Aug 2022 - Sep 2022, UGA \| Athens, GA)
    - Built *Next.js*/*TypeScript*/*Redux*/*Material-UI*/*styled-components*/*Framer Motion* application for visualizing over 30 sorting algorithms
    - Can add custom delay, change speed of animation, 5 different ways to provide input array, and multiple algorithms can be run in parallel
    - Deployed the application on Github Pages

- **3D Human Reconstruction** (Aug 2022, UGA \| Athens, GA)
    - Engineered a machine learning project to generate a full human body 3D object from random noise
    - Combined GAN, Pose Estimation, RGB-to-3D object models from 3 different repositories
    - Upgraded PyTorch dependency from 0.4.0, 1.4.0, 1.9.0 to 1.11.1 and upgraded all repositories to CUDA 11.3
    - Built *Next.js*/*TypeScript*/*ThreeJS* application to show the results and deployed it on Github Pages

- **Bookstore application** ()

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

### Blogs
All my blog posts can be found at [kushajveersingh.com/blog/](https://www.kushajveersingh.com/blog/).

### Certifications
- GCP Essentials (Qwiklabs) [link](https://www.cloudskillsboost.google/public_profiles/00544d10-99eb-435c-bba0-8099796ee428)
- Executive Data Science Specialization (Johns Hopkins University, Coursera)
    - A Crash Course in Data Science [link](https://www.coursera.org/account/accomplishments/verify/6MF9GPX5KUHP)
    - Building a Data Science Team [link](https://www.coursera.org/account/accomplishments/verify/MH8HYKF6WGAL)
    - Managing Data Analysis [link](https://www.coursera.org/account/accomplishments/verify/T74CCA5DKTJJ)
    - Data Science in Real Life [link](https://www.coursera.org/account/accomplishments/verify/H5F429NLE53V)
    - Executive Data Science Capstone [link](https://www.coursera.org/account/accomplishments/verify/PBML4NXBPF3L)
- Data Science at Scale Specialization (University of Washington, Coursera)
    - Data Manipulation at Scale: Systems and Algorithms [link](https://www.coursera.org/account/accomplishments/verify/C6GTQA5FED8B)
    - Practical Predictive Analytics: Models and Methods [link](https://www.coursera.org/account/accomplishments/verify/RF9GDHVJ9ZXT)
- Machine Learning (Stanford University, Coursera) [link](https://www.coursera.org/account/accomplishments/verify/BSQDLLWCXT4B)
- Bayesian Statistics: From Concept to Data Analysis (University of California, Santa Cruz, Coursera) [link](https://www.coursera.org/account/accomplishments/verify/UX2GCTWZK6RG)
- Neural Networks for Machine Learning (University of Toronto, Coursera) [link](https://www.coursera.org/account/accomplishments/verify/8CBSNM7BSPQ2)
- Deep Learning Specialization (DeepLearning.AI, Coursera)
    - Neural Networks and Deep Learning [link](https://www.coursera.org/account/accomplishments/verify/XXSA3B94FM5R)
    - Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization [link](https://www.coursera.org/account/accomplishments/verify/MHEC3N8QT9QH)
    - Structuring Machine Learning Projects [link](https://www.coursera.org/account/accomplishments/verify/X3SQXVCQRUCT)
    - Convolutional Neural Networks [link](https://www.coursera.org/account/accomplishments/verify/32B257T4VCDA)
    - Sequence Models [link](https://www.coursera.org/account/accomplishments/verify/GYTKJ9CPCK9S)
- Recommender Systems Specialization (University of Minnesota, Coursera)
    - Introduction to Recommender Systems: Non-Personalized and Content-Based [link](https://www.coursera.org/account/accomplishments/verify/WZTZJNCJJVPW)
    - Nearest Neighbor Collaborative Filtering [link](https://www.coursera.org/account/accomplishments/verify/CLZHGR8YVC2J)
    - Recommender Systems: Evaluation and Metrics [link](https://www.coursera.org/account/accomplishments/verify/4KZS9CQNZGYP)
    - Matrix Factorization and Advanced Techniques [link](https://www.coursera.org/account/accomplishments/verify/SS65M3NDKH2T)
- Genomic Data Science Specialization (Johns Hopkins University, Coursera)
    - Introduction to Genomic Technologies [link](https://www.coursera.org/account/accomplishments/verify/DQ69PEMKN58C)
    - Python for Genomic Data Science [link](https://www.coursera.org/account/accomplishments/verify/WDWT49566AQ6)
- Algorithms Specialization (Stanford, Coursera)
    - Divide and Conquer, Sorting and Searching, and Randomized Algorithms [[link](https://www.coursera.org/account/accomplishments/verify/P4BGEXSGJEJA)