import pandas as pd
import numpy as np
import torch as tc
import os
import shutil

from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms.v2 import Compose, ToDtype, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation
from gz2_dataset import PATH_GZ2_DATASET, GZ2_DATA_TABULAR_FILENAME
from image_preprocessing import PREPROCESS_IMAGE, PREPROCESS_IMAGE_NORMALIZE

# ==================================================
# ================ GLOBAL VARIABLES ================
# ==================================================

PATH_DATASET = "./dataset/"
PATH_DATASET_IMAGES = PATH_DATASET + "images/"
DATASET_DATA_TABULAR_FILENAME = "dataset.csv"
DATASET_SPLIT = [0.7, 0.15, 0.15]
GALAXY_TYPES = ["smooth-round", "smooth-in-between", "smooth-cigar", "disk-edgeon", "barred-spiral", "unbarred-spiral"]
NUM_CLASSES = len(GALAXY_TYPES)

# ==================================================
# ================= GALAXY DATASET =================
# ==================================================

class GalaxyDataset(Dataset):
    """A Galaxy Zoo 2's dataset that contains clean sample of galaxies."""

    def __init__(self, mode_dataset="train", device=tc.device("cpu")):
        if mode_dataset != "train" and mode_dataset != "valid" and mode_dataset != "test":
            raise ValueError("mode_dataset allows \"train\", \"valid\" or \"test\"")
        
        self._dataset = pd.read_csv(PATH_DATASET + DATASET_DATA_TABULAR_FILENAME)
        self._dataset_idxs = np.load(PATH_DATASET + "training_set.npy") if mode_dataset == "train" else \
                             np.load(PATH_DATASET + "validation_set.npy") if mode_dataset == "valid" else \
                             np.load(PATH_DATASET + "test_set.npy")
        self._transform = Compose([PREPROCESS_IMAGE, RandomHorizontalFlip(p=0.5), RandomVerticalFlip(p=0.5), RandomRotation(degrees=(0, 360)), ToDtype(dtype=tc.float32, scale=True)]) if mode_dataset == "train" else \
                          PREPROCESS_IMAGE_NORMALIZE
        self._device = device

    def __len__(self):
        return len(self._dataset_idxs)

    def __getitem__(self, index):
        point_idx = self._dataset_idxs[index]
        
        #Read image from disk.
        img = read_image(self._dataset["file-location"][point_idx] + self._dataset["filename"][point_idx])
        img = img.to(self._device)
        img = self._transform(img)

        #Get image's label.
        label = 0
        for i in range(len(GALAXY_TYPES)):
            if self._dataset[GALAXY_TYPES[i]][point_idx] == 1:
                label = i
                break
        
        return img, label

# ========================================
# ============= BUILD DATASET ============
# ========================================

if __name__ == "__main__":
    assert(np.sum(DATASET_SPLIT) == 1.0)

    print("A new dataset of galaxies are being created...")
    
    #Old dataset is deleted.
    if os.path.exists(PATH_DATASET):
        shutil.rmtree(PATH_DATASET)

    #Open Galaxy Zoo 2's data tabular.
    gz2_dataset = pd.read_csv(PATH_GZ2_DATASET + GZ2_DATA_TABULAR_FILENAME, delimiter=",")

    #Create new own dataset.
    os.makedirs(PATH_DATASET)
    entries = []

    for i in range(gz2_dataset.shape[0]):
        id_str               = str(format(gz2_dataset["dr7objid"][i]))
        filename             = "{}.jpg".format(gz2_dataset["dr7objid"][i])
        file_location        = PATH_GZ2_DATASET + "images/" + id_str[:6] + "/"
        is_smooth_round      = int(gz2_dataset["t01_smooth_or_features_a01_smooth_flag"][i] == 1            and gz2_dataset["t07_rounded_a16_completely_round_flag"][i] == 1)
        is_smooth_in_between = int(gz2_dataset["t01_smooth_or_features_a01_smooth_flag"][i] == 1            and gz2_dataset["t07_rounded_a17_in_between_flag"][i] == 1)
        is_smooth_cigar      = int(gz2_dataset["t01_smooth_or_features_a01_smooth_flag"][i] == 1            and gz2_dataset["t07_rounded_a18_cigar_shaped_flag"][i] == 1)
        is_lenticular_edgeon = int(gz2_dataset["t01_smooth_or_features_a02_features_or_disk_flag"][i] == 1  and gz2_dataset["t02_edgeon_a04_yes_flag"][i] == 1)
        is_barred_spiral     = int(gz2_dataset["t01_smooth_or_features_a02_features_or_disk_flag"][i] == 1  and gz2_dataset["t02_edgeon_a05_no_flag"][i] == 1                   and gz2_dataset["t04_spiral_a08_spiral_flag"][i] == 1       and gz2_dataset["t03_bar_a06_bar_flag"][i] == 1) 
        is_unbarred_spiral   = int(gz2_dataset["t01_smooth_or_features_a02_features_or_disk_flag"][i] == 1  and gz2_dataset["t02_edgeon_a05_no_flag"][i] == 1                   and gz2_dataset["t04_spiral_a08_spiral_flag"][i] == 1       and gz2_dataset["t03_bar_a07_no_bar_flag"][i] == 1) 

        if is_smooth_round + is_smooth_in_between + is_smooth_cigar + is_lenticular_edgeon + is_barred_spiral + is_unbarred_spiral == 1 and gz2_dataset["t06_odd_a15_no_flag"][i] and os.path.isfile(file_location+filename):
            entries.append([id_str, filename, file_location, is_smooth_round, is_smooth_in_between, is_smooth_cigar, is_lenticular_edgeon, is_barred_spiral, is_unbarred_spiral])

    dataset = pd.DataFrame(data=entries, columns=["id_str", "filename", "file-location"]+GALAXY_TYPES)
    dataset.to_csv(PATH_DATASET + DATASET_DATA_TABULAR_FILENAME)

    #Create new training, validation and test set.
    indices = np.arange(dataset.shape[0], step=1, dtype=np.int32)
    train_idxs = []
    valid_idxs = []
    test_idxs = []
    rng = np.random.default_rng()

    for class_name in GALAXY_TYPES:
        class_idxs = indices[(dataset[class_name] == 1).values]
        rng.shuffle(class_idxs)

        train_points_size = int(class_idxs.shape[0] * DATASET_SPLIT[0])
        for i in range(0, train_points_size):
            train_idxs.append(class_idxs[i])

        valid_points_size = int(class_idxs.shape[0] * DATASET_SPLIT[1])
        for i in range(train_points_size, train_points_size + valid_points_size):
            valid_idxs.append(class_idxs[i])

        for i in range(train_points_size + valid_points_size, class_idxs.shape[0]):
            test_idxs.append(class_idxs[i])

    np.save(PATH_DATASET + "training_set.npy", train_idxs)
    np.save(PATH_DATASET + "validation_set.npy", valid_idxs)
    np.save(PATH_DATASET + "test_set.npy", test_idxs)

    #Print a summary.
    print("----------------------------------------------------------------------------------------------------")
    n_galaxy_types = np.zeros(len(GALAXY_TYPES), dtype=np.float32)
    for i in range(len(GALAXY_TYPES)):
        n_galaxy_types[i] = np.sum(dataset[GALAXY_TYPES[i]].values)

    summary_dataset = pd.DataFrame(data=[n_galaxy_types.astype(dtype=np.int32).tolist(), (n_galaxy_types / dataset.shape[0]).tolist()],
                                   columns=GALAXY_TYPES,
                                   index=["total", "portion"])
    print(summary_dataset)
    print("----------------------------------------------------------------------------------------------------")

    print("New dataset created.")


# ----------------------------------------------------------------------------------------------------
#          smooth-round  smooth-in-between  smooth-cigar  disk-edgeon  barred-spiral  unbarred-spiral
# total    10422.000000       11137.000000   1156.000000  6079.000000    4320.000000      6633.000000
# portion      0.262208           0.280197      0.029084     0.152942       0.108687         0.166881
# ----------------------------------------------------------------------------------------------------