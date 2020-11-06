# AI6126 Project 1: CelebA Facial Attribute Recognition Challenge

## Challenge Description
The goal of this mini challenge is to identify the attribute label depicted in a facial
photograph. The data for this task comes from the CelebA dataset [[1]], which contains
200 thousand images belonging to 40 attribute labels. Specifically, the challenge data for
this course consists of 160,000 images for training, 20,000 images for validation and
20,000 images for testing. The images will be pre-cropped and aligned to make the data
more manageable.

For each image, algorithms will produce a list of all possible attribute labels. The quality
of a labeling will be evaluated based on the label that best matches the ground truth label
for the image. The idea is to allow an algorithm to identify multiple attribute labels in an
image given that humans often describe a face using different words (e.g. black hair, big
eyes, smiling).

## Submission Guideline
Students should improve the classification accuracy of their network models.
-[x] Download dataset [[1]], use the images in “img_align_celeba.zip” as well as the attribute labels.
-[x] Train your network using the training set of CelebA.
-[x] Tune the hyper-parameters using the validation set of CelebA.
-[x] Submit predictions of the test set for evaluations and ranking in the mini challenge leaderboard. The test set will be available one week before the deadline.
-[x] No external data are allowed in this mini challenge. Only ImageNet pre-trained models are allowed.
-[x] You should not use an ensemble of models.

## Submission Items
-[x] Short report not more than five A4 pages (Arial 10 font) to describe:
    - [x] Model used
    - [x] Loss functions
    - [x] Any processing or operations used to obtain results
    - [x Accuracy of each attribute and average accuracy obtained
-[x] Folder containing:
    - [x] Predictions of the test set
    - [x] Codes for training and testing of model
-[x] Readme.txt containing:
    - [x] Description of the files you have submitted
    - [x] References to the third party libraries
    - [x] Details to run and test solution
    
## Tips
Refer to [[2]] to get started.
Techniques to improve recognition accuracy:
+ Data augmentation e.g. random flip [3] 
+ Deeper model e.g ResNet-50 [4]
+ Advanced loss functions e.g. focal loss [5]

## References 
[1]: <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>
[2]: https://github.com/d-li14/face-attribute-prediction
\[1\]: Z. Liu et al. Deep Learning Face Attributes in the Wild, ICCV 2015

\[2\]: Face attribute prediction 

[3]: He et al. Bag of Tricks for Image Classification with Convolutional Neural Networks,
ArXiv 2018

[4]: He et al. Deep Residual Learning for Image Recognition, CVPR 2016

[5]: T-Y Lin et al., Focal Loss for Dense Object Detection, ICCV 2017


