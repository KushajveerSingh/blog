---
layout: page
title: Projects
permalink: /projects/
---

### Table of contents
- [Education](#education)
- [Work Experience](#work-experience)
- [Additional Experience](#additional-experience)
- [Projects](#projects)
- [Blogs](#blogs)
- [Certifications](#certifications)

### Education
- **Master of Science** in Computer Science
  - University of Georgia, Athens, GA, USA (Jan 2021 - Dec 2022)
  - GPA: 3.82 / 4.00

- **Bachelor of Technology** in Electronics and Communication Engineering 
  - Punjab Engineering College (Deemed to be University), Chandigarh, India (Aug 2016 - Jun 2020)
  - CGPA: 7.0 / 10.0

### Work Experience
- **Full Stack Developer** (Jul 2022 - current, UGA)
    - Worked in the IT department of the College of Agriculture and Environmental Sciences, University of Georgia (Athens, GA, USA).
    - Migrated a legacy *Microsoft Access* database to *PostgreSQL* with 18 years of data in it.
    - Created a parser in *Node.js* and *Express* to convert *Access* queries to *PostgreSQL* queries on the fly to maintain the existing 217 programs.
    - Created the development and production servers, which included doing IP whitelisting, backup configuration with a local storage server, and periodically syncing development and production servers.
    - To prevent people from accessing the SQL queries from outside, built a program in *Node.js* using which queries can be written in a text file, that is never shared with the public, and refer to them using a special function in *JavaScript*.
    - Automated the data entry pipeline using *Python* and *JavaScript*, resulting in a reduction of time from 2 minutes per handwritten form to 25 seconds on average. This was done by implementing autocomplete functionality for various fields, like name, address, contact info and merging it with the existing database interface. 
    - Used *Selenium* to automate browser-related tasks, like downloading an invoice from UPS and downloading reports.
    - Used *Next.js*/*TypeScript*/*Redux*/*Material-UI* to convert 3 existing programs from *JavaScript*/*jQuery*
    - Rewrote the entire codebase for a proprietary language, which included converting information from a 200-page manual to a *Next.js* application and updating the queries from *Access* to *PostgreSQL*.

### Additional Experience
- **Google CS Research Mentorship Program** (Sep 2022 - Dec 2022)
    - Selected for a 12-week competitive Google CS Research Mentorship Program (CSRMP 2022B) among 50 candidates selected from all across USA and Canada.
    - Google-to-student mentorship program that inspires students from Historically Marginalized Groups (HMGs) to pursue and persist in CS research careers.

- **Poster submission, School of Computing Research Day** (Sep 2022 - Oct 2022)
    - Submitted 2 posters for the School of Computing, Research Day at University of Georgia (Athens, GA, USA)
    - Created a module that improves the robustness of Graph Neural Networks (GNNs) against adversarial attacks
    - Proposed module beats state-of-the-art in 3 out of 4 datasets, while only adding 5 lines of code
    - Presented a poster on generating full human body 3D scans from random noise

- **Graduate Teaching Assistant** (Jan 2022 - Aug 2022, UGA)
    - Worked in the Computer Science department, University of Georgia (Athens, GA, USA).
    - Led weekly meetings and facilitated group discussions for the course *Discrete Mathematics for Computer Science*
    - Held one-on-one office hours for 2 transfer students, covering the entire course in detail and discussing practical applications of the content

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
- **Youtube Video Platform** [github](https://github.com/KushajveerSingh/youtube_video_platform), [demo](https://kushaj-youtube-video.vercel.app/)
  - Built a fully responsive *React*/*Next.js*/*Material-UI* web app using Youtube API and deployed on *Vercel*
  - Implemented video section, category section, responsive channel and video cards, channel pages, video pages with ability to play videos and see related videos

- **Group Video Chat** [github](https://github.com/KushajveerSingh/video-chat-app), [demo](https://kushaj-video-chat.vercel.app)
  - Built a group video chat app built using *Agora*, and deployed the backend on *Heroku* and frontend on *Vercel*.
  - Conference chat with multiple people at same time, screen sharing capabilities, private and group messaging, admin controls, create polls
  - Worked with an enterprise size codebase and added features on top of it

- **Chat Messaging App** (WORK IN PROGRESS)
  - Built a fully-responsive full-stack realtime chat messaging application 
  - Support for authentication, Twilio SMS notification, direct and group chats, emojis and reactions, GIF support, edit/delete messages
  - Built *React* frontend using *Stream Chat* API and deployed on *Vercel*
  - Built *Express* backend and deployed on *Heroku*

- **Sort Visualizer** [github](https://github.com/KushajveerSingh/sort_visualizer), [demo](https://kushaj-sort-visualizer.vercel.app/)
  - Built *React*/*Redux*/*Material-UI* web application for visualizing sorting algorithms and deployed on *Vercel*
  - Implemented over 30 sorting algorithms, with ability to view upto 9 algorithms in parallel

- **3D Human Reconstruction** [github](https://github.com/KushajveerSingh/human_3d_reconstruction)
    - Engineered a machine learning project to generate a full human body 3D object from random noise
    - Combined GAN, Pose Estimation, RGB-to-3D object models from 3 different repositories
    - Upgraded PyTorch dependency from 0.4.0, 1.4.0, 1.9.0 to 1.11.1 and upgraded all repositories to CUDA 11.3 from 10.x versions

- **Bookstore application** [github](https://github.com/KushajveerSingh/genlib)
  - Built a full-stack bookstore application using *Django*/*SQLite*
  - Implemented user authentication with email, change/forget password, edit profile, advanced search options, cart management and placing an order

- **JSON/YAML Graph Visualizer** (WORK IN PROGRESS)
    - Build *Next.js*/*Redux*/*Material-UI* application to convert JSON/YAML data into a graph for improved readability deployed on *Vercel*
    - Wrote a parser in *TypeScript* to convert YAML to JSON

- **Credit Assessment** [github](https://github.com/KushajveerSingh/ds_cup)
  - Created a fair and explainable model to approve credit card requests
  - Used *LIME* predictions and built *PySimpleGUI* to explain the predictions

- **Resizer Network for Computer Vision** [github](https://github.com/KushajveerSingh/resize_network_cv), [blog](https://www.kushajveersingh.com/blog/data-augmentation-with-resizer-network-for-image-classification)
  - Built *PyTorch* model to resize images for downstream tasks, based on Google AI *Learning to Resize Images for Computer VIsion Tasks*
  - Tested model on 2 subsets of ImageNet dataset and demonstrated the improved performance using the proposed model

- **Parking Lot Detection** [github](https://github.com/KushajveerSingh/Unsupervised-Parking-Lot-Detection)
  - Built a fully deployable and unsupervised parking space detection system using *PyTorch*
  - Ability to adjust the predictions based on where cars are parked over time
  - Combine results from multiple frames to fill spots and make the predictions more robust

- **Semantic Image Synthesis** [github](https://github.com/KushajveerSingh/SPADE-PyTorch), [blog](https://www.kushajveersingh.com/blog/spade-state-of-the-art-in-image-to-image-translation-by-nvidia)
  - Open-sourced the first public implementation of GauGAN paper by Nvidia
  - Implemented *PyTorch* GAN model to convert an image map into a realistic image

- **Style Transfer** [github](https://github.com/KushajveerSingh/Photorealistic-Style-Transfer), [blog](https://www.kushajveersingh.com/blog/all-you-need-for-photorealistic-style-transfer-in-pytorch)
  - Implemented *PyTorch* model to transfer style between images

- **Unscramble Game Solver** [github](https://github.com/KushajveerSingh/deep_learning/tree/master/random/unscramble_android_game)
  - Created a *Python* program to solve *Unscramble* android game
  - Implemented an efficient English dictionary lookup

- **Random Duty List** [github](https://github.com/KushajveerSingh/Random-Duty-List)
  - Built PHP/MySQL program for Chandigarh, India police department
  - Assign duties at various stations without repetition between days


### Jupyter Notebooks

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


* **Waste Seggregation using trashnet** [github](https://github.com/KushajveerSingh/deep_learning/tree/master/projects/Waste_Seggregation_using_trashnet). Contains the code to train models for trashnet and then export them using ONNX. It was part of a bigger project where we ran these models on Rasberry Pi, which controlled wooden planks to classify the waste into different categories (code for rasberry pi not included here).


### Blogs
- Complete tutorial on building images using Docker [link](https://www.kushajveersingh.com/blog/docker)
- Data augmentation with learnable Resizer network for Image Classification [link](https://www.kushajveersingh.com/blog/data-augmentation-with-resizer-network-for-image-classification)
- Writing custom CUDA kernels with Triton [link](https://www.kushajveersingh.com/blog/writing-custom-cuda-kernels-with-triton)
- Complete tutorial on how to use Hydra in Machine Learning projects [link](https://www.kushajveersingh.com/blog/complete-tutorial-on-how-to-use-hydra-in-machine-learning-projects)
- What can neural networks reason about? [link](https://www.kushajveersingh.com/blog/what-can-neural-networks-reason-about)
- ImageNet Dataset Advancements [link](https://www.kushajveersingh.com/blog/imagenet-dataset-advancements)
- Deep Learning Model Initialization in Detail [link](https://www.kushajveersingh.com/blog/deep-learning-model-initialization-in-detail)
- How to setup personal blog using Ghost and Github hosting [link](https://www.kushajveersingh.com/blog/how-to-setup-personal-blog-using-ghost-and-github-hosting)
- Study of Mish activation function in transfer learning with code and discussion [link](https://www.kushajveersingh.com/blog/study-of-mish-activation-function-in-transfer-learning)
- Reproducing Cyclic Learning papers + SuperConvergence using fastai [link](https://www.kushajveersingh.com/blog/reproducing-cyclic-learning-papers-and-superconvergence)
- How to become an expert in NLP in 2019 [link](https://www.kushajveersingh.com/blog/how-to-become-an-expert-in-nlp-in-2019)
- All you need for Photorealistic Style Transfer in PyTorch [link](https://www.kushajveersingh.com/blog/all-you-need-for-photorealistic-style-transfer-in-pytorch)
- SPADE: State of the art in Image-to-Image Translation by Nvidia [link](https://www.kushajveersingh.com/blog/spade-state-of-the-art-in-image-to-image-translation-by-nvidia)
- Weight Standardization: A new normalization in town [link](https://www.kushajveersingh.com/blog/weight-standardization-a-new-normalization-in-town)
- Training AlexNet with tips and checks on how to train CNNs: Practical CNNs in PyTorch [link](https://www.kushajveersingh.com/blog/training-alexnet-with-tips-and-checks-on-how-train-cnns-practical-cnns-in-pytorch)
- Theoretical Machine Learning: Probabilities and Statistical Math [link](https://www.kushajveersingh.com/blog/theoretical-machine-learning-probalistic-and-statistical-math)

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
    - Divide and Conquer, Sorting and Searching, and Randomized Algorithms [link](https://www.coursera.org/account/accomplishments/verify/P4BGEXSGJEJA)