import cv2
import pandas as pd
import numpy as np
import torch as tc

from torchvision.transforms.v2 import Compose, CenterCrop, Resize, ToDtype
from torchvision.transforms.v2.functional import to_image, to_pil_image, center_crop, resize, adjust_brightness, to_dtype, normalize, horizontal_flip, rotate

# ========================================
# ========== PREPROCCESING IMAGE =========
# ========================================

PREPROCESS_IMAGE = Compose([CenterCrop(256), Resize(128, antialias=True)])
PREPROCESS_IMAGE_NORMALIZE = Compose([PREPROCESS_IMAGE, ToDtype(dtype=tc.float32, scale=True)])

# ========================================
# ===== PREPROCESSING TEST FUNCTIONS =====
# ========================================

def preprocess_1(img):
    img = to_image(img)
    img = resize(img, [128, 128], antialias=True)
    img = to_pil_image(img)
    return np.asarray(img)

def preprocess_2(img):
    img = to_image(img)
    img = resize(img, [212, 212], antialias=True)
    img = center_crop(img, [128, 128])
    img = to_pil_image(img)
    return np.asarray(img)

def preprocess_3(img):
    img = to_image(img)
    img = center_crop(img, [296, 296])
    img = resize(img, [128, 128], antialias=True)
    img = to_pil_image(img)
    return np.asarray(img)

def preprocess_4(img):
    img = to_image(img)
    img = center_crop(img, [256, 256])
    img = resize(img, [128, 128], antialias=True)
    img = to_pil_image(img)
    return np.asarray(img)

def preprocess_5(img):
    img = to_image(img)
    img = center_crop(img, [256, 256])
    img = resize(img, [128, 128], antialias=True)
    #img = adjust_brightness(img, 1)
    img = to_dtype(img, scale=True)
    img = normalize(img, mean=[0.2, 0.2, 0.2], std=[0.5, 0.5, 0.5])
    img = to_pil_image(img)
    img = 255 * np.asarray(img)
    return img

# ========================================
# ====== DATA-AUGMENTATION FUNCTION ======
# ========================================

def data_agmentation(img):
    img = to_image(img)
    
    if np.random.default_rng().uniform(0, 1) <= 0.5:
        img = horizontal_flip(img)

    img = rotate(img, np.random.default_rng().uniform(0, 180))
    img = to_pil_image(img)
    return np.asarray(img)

# ========================================
# =========== TESTING FUNCTIONS ==========
# ========================================

if __name__ == "__main__":
    from dataset import PATH_DATASET, DATASET_DATA_TABULAR_FILENAME

    USE_DATA_AUGMENTATION = False
    N_SAMPLES = 20
    dataset = pd.read_csv(PATH_DATASET + DATASET_DATA_TABULAR_FILENAME)
    preprocessing_functions = [preprocess_1, preprocess_2, preprocess_3, preprocess_4, preprocess_5]

    idxs_sampled = np.random.default_rng().choice(dataset.shape[0], size=N_SAMPLES, replace=False)

    for idx in idxs_sampled:
        img = cv2.imread(dataset["file-location"][idx] + dataset["filename"][idx])

        cv2.imshow("original", img)
        for preprocess_function in preprocessing_functions:
            img_edit = preprocess_function(img)
            if USE_DATA_AUGMENTATION:
                img = data_agmentation(img)

            cv2.imshow(preprocess_function.__name__, img_edit)

        cv2.waitKey(0)