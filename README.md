# DL-Galaxy-Classification

## About the project
This is a Deep Learning project that aims to classify galaxies based on own morphology properties with use of Xception neural network.

## Libraries <a name="libs"></a>
The following libraries used for this project are:
- torch v.2.1.2
- torchvision v.0.16.2
- scikit-learn v.1.5.1
- numpy v.1.26.3
- pandas v.2.2.2
- opencv-python v.4.9.0.80
- [galaxy-datasets](https://pypi.org/project/galaxy-datasets/) v.0.0.21
- [torchsummary](https://github.com/sksq96/pytorch-summary) v.1.5.1

## How to setup the project
Before you use this project, you need to do the following steps to setup the project:
1. you need to download libraries required from project to be run and you can find them on section [Libraries](#libs).
   It shouldn't be needed to download same version of libraries and it don't need to download "torchsummary" library because
   the goal of this library is to display summary information of a neural networks (such as its architecture).
2. download the project's source code and extract it.
3. the dataset isn't included and you need to download it. The dataset used on this project is called [Galaxy Zoo 2](https://arxiv.org/abs/1308.3496v2)
   and it is made up from a csv file called [gz2_hart16.csv](https://gz2hart.s3.amazonaws.com/gz2_hart16.csv.gz)
   and galaxy images that you can downlaod running [gz2_dataset.py](https://github.com/bottamichele/DL-Galaxy-Classification/blob/main/gz2_dataset.py) setting DOWNLOAD_NEEDED to True.
   You can find Galaxy Zoo 2's dataset on [Galaxy Zoo Data](https://data.galaxyzoo.org/).
4. when you downloaded the dataset, put gz2_hart16.csv on "gz2_dataset" folder located at project folder. 
5. when gz2_hart16.csv and galaxy images are on "gz2_dataset" folder and if you want to do training and testing of a model,
   you need to run [dataset.py](https://github.com/bottamichele/DL-Galaxy-Classification/blob/main/dataset.py)
   to create new dataset with clean sample and with training, validation and test set togheter.
