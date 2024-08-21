# DL-Galaxy-Classification

## About project

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

## Requirements
Before you use this project, you need to do the following steps:
1. you need to download libraries required from project to be run and you can find them on section [Libraries](#libs).
   It shouldn't be needed to download same version of libraries and it don't need to download "torchsummary" library because
   the goal of this library is to display summary information of a neural networks (such as its architecture).
2. the dataset isn't included and you need to download it. The dataset used on this project is called Galaxy Zoo 2,
   that can be dowloaded on [Galaxy Zoo Data](https://data.galaxyzoo.org/), and it is made up from a csv file called [gz2_hart16.csv](https://gz2hart.s3.amazonaws.com/gz2_hart16.csv.gz)
   and galaxy images, can be downlaoded running [gz2_dataset.py](https://github.com/bottamichele/DL-Galaxy-Classification/blob/main/gz2_dataset.py) setting DOWNLOAD_NEEDED to True.
3. When you downloaded the dataset, put gz2_hart16.csv on "gz2_dataset" folder.
4. 
