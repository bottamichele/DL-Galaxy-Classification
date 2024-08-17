from dataset import PATH_DATASET, DATASET_DATA_TABULAR_FILENAME, PATH_DATASET_IMAGES, GALAXY_TYPES

import pandas as pd
import shutil
import os

if __name__ == "__main__":
    #Load data tabular from disk.
    dataset = pd.read_csv(PATH_DATASET + DATASET_DATA_TABULAR_FILENAME)

    #Create folders of each class type galaxy.
    for name in GALAXY_TYPES: 
        os.makedirs(PATH_DATASET_IMAGES + name + "/", exist_ok=True)

    #Copy galaxy images
    for i in range(dataset.shape[0]):
        for type_name in GALAXY_TYPES:
            if dataset[type_name][i] == 1:
                class_galaxy = type_name

        shutil.copy2(dataset["file-location"][i]+dataset["filename"][i], PATH_DATASET_IMAGES + class_galaxy + "/" + dataset["filename"][i])