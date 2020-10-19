# Source Codes
## Submission Files

```
​```
data
|
└───celeba                                                       //public Celeba dataset
	└───img_align_celeba										 //directory containing public dataset images
	└───list_attr_celeba.txt
	└───list_eval_partition.txt
	└───train_attr_list.txt
	└───test_attr_list.txt
	└───val_attr_list.txt
└───testset                                                      //private Celeba testset
src
│   README.md
│   ai6126-p1-train-v1.X.ipynb                                   //notebook for training the model
|   ai6126-p1-inference-v1.X.ipynb                               //notebook for performing inference
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
└───infs 														 //directory to keep models for inference   
​```
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

## How to perform

### Training

1. Setup environment.
2. Open the `ai6126-p1-train-v1.X.ipynb` Jupyter notebook.
3. Update the configurations in `config.py` to desired settings.
4. Click on "Run All Cells".
5. Output model will be saved into "\backups" folder.
6. Alternatively, you may `Download As` python `.py` file and run using `python ai6126-p1-train-v1.X.py ` .

### Inference

1. Setup environment.
2. Ensure there is at least one trained model folder in "\inf" directory. (usually copied from "\backups")
3. Open the `ai6126-p1-inference-v1.X.ipynb` Jupyter notebook.
4. Update the configurations in `config.py` to desired settings.
5. Click on "Run All Cells".
6. Output predictions.csv and accuracies.json will be saved into the model folder in "\inf" directory.
7. Alternatively, you may `Download As` python `.py` file and run using `python ai6126-p1-inference-v1.X.py ` .
8. (Optional) Analyze predictions using `P_Predictions Analysis.ipynb` notebook.

## References

<u>Disclaimer</u>: Most adapted codes are already credited within the source codes. 

Here are to mention some of the referenced codes:

+ Baseline codes: https://github.com/d-li14/face-attribute-prediction

+ MixUp & LabelSmoothingCrossEntropy codes: modified from [fast.ai](https://www.fast.ai/) course

+ FocalLoss codes: modified from [Kornia](https://github.com/kornia/kornia/blob/master/kornia/losses/focal.py), a differentiable computer vision library for PyTorch.
+ CosineWarmupLR codes: from [torch-toolbox](https://github.com/PistonY/torch-toolbox/blob/master/torchtoolbox/optimizer/lr_scheduler.py).

