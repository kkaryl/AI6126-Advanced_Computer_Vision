# AI6126: Assignment 1

## Problem Overview

The goal of this mini challenge is to identify the attribute label depicted in a facial photograph. The data for this task comes from the CelebA dataset [[1]], which contains 200 thousand images belonging to 40 attribute labels. Specifically, the challenge data for this course consists of 160,000 images for training, 20,000 images for validation and 20,000 images for testing. The images will be pre-cropped and aligned to make the data more manageable.

For each image, algorithms will produce a list of all possible attribute labels. The quality of a labeling will be evaluated based on the label that best matches the ground truth label for the image. The idea is to allow an algorithm to identify multiple attribute labels in an image given that humans often describe a face using different words (e.g. black hair, big eyes, smiling).

<u>Constraints</u>

+ Model training strictly on CelebA training set.
+ ImageNet pre-trained model only.
+ Ensemble of models is not allowed.

<u>Assessment Criteria</u>

Evaluate and rank the performance of each submitted solution based on the
average accuracy across all attributes on a private test set. The higher your prediction accuracy is, the higher the score you will receive.

## Final Model

MobileNetV2 with random horizontal flip, random affine transformations (shift, scale and rotate) as well as MixUp training using Focal Loss.

Note: [Model answer](https://github.com/NIRVANALAN/face-attribute-prediction) was released by professor after grading. The private testset was announced to be from [lfwA+](vis-www.cs.umass.edu/lfw/). 

| Model           | Train Accuracy | Validation Accuracy | Test Accuracy        |
| --------------- | -------------- | ------------------- | -------------------- |
| **MobileNetV2** | 91.55          | 92.28               | 91.71                |
| ResNeXt50       | 92.88          | 92.10               | 91.65                |
| Model Answer 1  | --             | --                  | 92.14 (73.31 LFWA)   |
| Model Answer 2  | --             | --                  | 91.7 (**73.9** LFWA) |

## Submission Files

```
predictions.csv													 //predictions of Celeba private testset
predictions.txt													 //predictions of Celeba private testset txt form
ai6126acv_p1.yml												 //conda environment setup file
data
|
└───celeba                                                       //public Celeba dataset
	└───img_align_celeba										 //directory containing public dataset images
	└───list_attr_celeba.txt
	└───list_eval_partition.txt
	└───train_attr_list.txt
	└───test_attr_list.txt
	└───val_attr_list.txt
	└───train_val_test_split.py
└───testset                                                      //private Celeba testset
|
src
│   README.md
│   ai6126-p1-train-v1.X.ipynb                                   //notebook for training the model
|   ai6126-p1-inference-v0.X.ipynb                               //notebook for performing inference
|   celeba_dataset.py                                            //custom dataset for Celeba
|   config.py                                                    //common configuration file for all notebooks
|   EDA_Celeba.ipynb											 //notebook used to perform EDA
|   EDA_Check Duplicates.ipynb									 //notebook used to check for duplicate images
|   DA_Test Transforms.ipynb									 //notebook for visualizing transforms
|   DA_MixUp Testing.ipynb										 //notebook used to test MixUp Training
|   duplicated.json                                              //json dictionary of duplicated images in Celeba
|   P_Predictions Analysis.ipynb								 //notebook for additional prediction analytics
|   
│
└───losses                                                       //directory containing all loss functions
      └───FocalLoss.py
      └───LabelSmoothingCrossEntropy.py
      └───loss_utils.py
      └───MixedUp.py
│
└───models                                                       //directory containing model files
      └───face_attr_net.py
│
└───utils  														 //directory containing helper functions
      └───bag_of_tricks.py										 //codes for no_bias_decay and mixup
      └───logger.py												 //file logger
      └───model_timer.py										 //timer class for tracking train times
      └───train_functions.py								     //codes for various training functions
│
└───infs 														 //dir to copy trained model folder for inference
	└─── MODELFOLDER
```

## Dependencies

### Base Environment

+ PyTorch 1.6.0
+ Torchvision 0.7.0
+ Tensorboard 2.3.0
+ MatPlotLib 3.3.1
+ Seaborn 0.11.0
+ Tqdm 4.49.0
+ Pandas 1.1.0
+ Numpy 1.19.1
+ Jupyter 

### Third Party

+ [Albumentations 0.4.6](https://github.com/albumentations-team/albumentations)
+ Imgaug 0.4.0 
+ Pillow 7.2.0
+ OpenCV 4.0.1 

### Anaconda setup

```shell script
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
conda install matplotlib
conda install seaborn
conda install pandas
conda install tqdm
conda install jupyter
conda install -c conda-forge jupyterlab
conda install -c conda-forge tensorboard
conda install -c conda-forge protobuf # for tensorboard
conda install -c conda-forge tensorboardx # if tensorboard does not work
conda install nb_conda_kernels # auto add kernels

# Third party APIs
conda install -c conda-forge imgaug
conda install albumentations -c conda-forge
```

Alternatively, you may install the environment using the `.yml` file provided.

```shell
conda env create --file ai6126acv_p1.yml
```

## How to perform

### Training

1. Setup environment and copy dataset to corresponding `..\Data` folder.
2. Open the `ai6126-p1-train-v1.X.ipynb` Jupyter notebook.
3. Update the configurations in `config.py` to desired settings.
4. Click on "Run All Cells".
5. Output model will be saved into "\backups" folder.
6. Alternatively, you may `Download As` python `.py` file and run using `python ai6126-p1-train-v1.X.py ` .

### Inference

1. Setup environment and copy dataset to corresponding `..\Data` folder.
2. Ensure there is at least one trained model folder in "\inf" directory. (usually copied from "\backups")
3. Open the `ai6126-p1-inference-v0.X.ipynb` Jupyter notebook.
4. Update the configurations in `config.py` to desired settings.
5. Click on "Run All Cells".
6. Output predictions.csv, predictions.txt and accuracies.json will be saved into the model folder in "\inf" directory.
7. Alternatively, you may `Download As` python `.py` file and run using `python ai6126-p1-inference-v0.X.py ` .
8. (Optional) Analyze predictions using `P_Predictions Analysis.ipynb` notebook.

## References

<u>Disclaimer</u>: Most adapted codes are already credited within the source codes. 

Here are to mention some of the referenced codes:

+ Baseline codes: https://github.com/d-li14/face-attribute-prediction

+ MixUp & LabelSmoothingCrossEntropy codes: modified from [fast.ai](https://www.fast.ai/) course

+ FocalLoss codes: modified from [Kornia](https://github.com/kornia/kornia/blob/master/kornia/losses/focal.py), a differentiable computer vision library for PyTorch.

