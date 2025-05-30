# DL-Galaxy-Classification

## About the project
This is a Deep Learning project that aims to classify galaxies based on their morphology properties with the use of the Xception neural network.

## Libraries <a name="libs"></a>
The following libraries were used for this project:
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
1. you need to download the libraries required for the project to run; you can find them in the [Libraries](#libs) section.
   It shouldn't be necessary to download the same version of libraries and to download the "torchsummary" library because
   its goal is to display summary information about neural networks (such as their architecture).
2. download the project's source code and extract it.
3. the dataset isn't included and you need to download it. The dataset used on this project is called [Galaxy Zoo 2](https://arxiv.org/abs/1308.3496v2)
   and it is made up of a csv file called [gz2_hart16.csv](https://gz2hart.s3.amazonaws.com/gz2_hart16.csv.gz)
   and galaxy images that you can downlaod by running [gz2_dataset.py](https://github.com/bottamichele/DL-Galaxy-Classification/blob/main/gz2_dataset.py) setting DOWNLOAD_NEEDED to True.
   You can find the Galaxy Zoo 2's dataset on [Galaxy Zoo Data](https://data.galaxyzoo.org/).
4. when you have downloaded the dataset, put gz2_hart16.csv in the "gz2_dataset" folder located in the project folder. 
5. when gz2_hart16.csv and the galaxy images are in the "gz2_dataset" folder and if you want to train and test a model,
   you need to run [dataset.py](https://github.com/bottamichele/DL-Galaxy-Classification/blob/main/dataset.py)
   to create new dataset with clean sample and with training, validation and test sets together.

## Models trained
The trained models of the project are available and can be found on the [Releases](https://github.com/bottamichele/DL-Galaxy-Classification/releases) page of this repository.
When you use one of these trained models, make sure they are in the "models" folder within the project folder.
