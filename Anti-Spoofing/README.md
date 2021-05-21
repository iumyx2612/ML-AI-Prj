# Face Anti Spoofing
## Description
This is a project I got from my school lab  
Using **Convolutional Neural Network** instead of hand crafted features to prevent face spoofing attack  
## Dataset
The dataset I used contains images from [NUAA Imposter Database](http://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/NUAAImposterDB_download.htm)
and a extrated face images from [CelebA Spoof](https://github.com/Davidzhangyuanhan/CelebA-Spoof)
The data folder should look like this if you want to train the model on your own data  
📦Data  
 ┣ 📂Real  
 ┃ ┣ 0.jpg  
 ┃ ┣ 1.jpg  
 ┃ ┣ 2.jpg  
 ┣ 📂Spoof  
 ┃ ┣ 0.jpg  
 ┃ ┣ 1.jpg  
 ┃ ┣ 2.jpg  
## Model
The model is defined in training/LivenessNet.py  
My model base on [pyimagesearch](https://www.pyimagesearch.com/2019/03/11/liveness-detection-with-opencv/) model with little adjustment  
The model accepts images with various size, currently I'm training the model with 64x64 images  
## Metrics
This is a binary classification problem so I'm currently using these 2 metrics
- TPR (True positive rate)
- FPR (False positive rate)
## Progress
- [x] Extract faces from NUAA Dataset and preprocess
- [x] Create model
- [x] Train model with aprx 20k images
- [x] Metrics 
- [ ] Compare model with other models 
