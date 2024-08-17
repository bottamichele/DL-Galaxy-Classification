import numpy as np
import cv2
import os
import shutil

from torch.nn.functional import relu

from torchvision.io import read_image
from torchvision.utils import make_grid, save_image
from torchvision.transforms.v2.functional import to_pil_image

from xception import Xception
from dataset import GALAXY_TYPES, NUM_CLASSES
from image_preprocessing import PREPROCESS_IMAGE_NORMALIZE

def save_feature_map(img, n_rows, filename):
    img = img.view(size=(img.size(1), 1, img.size(2), img.size(3)))
    save_image(make_grid(img, n_rows, padding=2, normalize=True), filename)

if __name__ == "__main__":
    SHOW_FEATURE_MAPS = False
    PATH_FMAPS = "./xception_fmaps/"
    IMAGE_NAME = "./gz2_dataset/images/588013/588013381662081125.jpg"

    #Load a Xception's model trained.
    model = Xception(NUM_CLASSES)
    model.load_model("./model/xception_model.pth")
    model.eval()

    #Load a galaxy image from disk.
    img = read_image(IMAGE_NAME)
    res_img = img.view(size=(1, img.size(0), img.size(1), img.size(2)))
    res_img = PREPROCESS_IMAGE_NORMALIZE(res_img)

    #The Xception's model classifies the galaxy image choosen.
    out = model(res_img)

    print(f"- Galaxy Type = {GALAXY_TYPES[out.argmax(dim=1).item()]}")
    print(f"- Output model = {out.detach().numpy()}")

    #Display galaxy image.
    cv2_img = cv2.cvtColor(np.asarray(to_pil_image(img)), cv2.COLOR_RGB2BGR)
    cv2.imshow("Galaxy Image", cv2_img)
    cv2.waitKey(0)

    #Show feature maps.
    if SHOW_FEATURE_MAPS:
        if os.path.exists(PATH_FMAPS):
            shutil.rmtree(PATH_FMAPS)
        os.makedirs(PATH_FMAPS)

        save_feature_map(res_img, 1, PATH_FMAPS + "0_galaxy.jpg")

        res = model._conv_1(res_img)
        save_feature_map(res, 6, PATH_FMAPS + "1_conv_1.jpg")

        res = model._conv_2(res)
        save_feature_map(res, 8, PATH_FMAPS + "2_conv_2.jpg")

        res = model._block_1(res)
        save_feature_map(res, 12, PATH_FMAPS + "3_block_1.jpg")
        save_feature_map(relu(res), 12, PATH_FMAPS + "3_block_1_relu.jpg")

        res = model._block_2(res)
        save_feature_map(res, 16, PATH_FMAPS + "4_block_2.jpg")
        save_feature_map(relu(res), 16, PATH_FMAPS + "4_block_2_relu.jpg")

        res = model._block_3(res)
        save_feature_map(res, 27, PATH_FMAPS + "5_block_3.jpg")
        save_feature_map(relu(res), 27, PATH_FMAPS + "5_block_3_relu.jpg")

        res = model._block_4_11[0](res)
        save_feature_map(res, 27, PATH_FMAPS + "6_block_4.jpg")
        save_feature_map(relu(res), 27, PATH_FMAPS + "6_block_4_relu.jpg")

        res = model._block_4_11[1](res)
        save_feature_map(res, 27, PATH_FMAPS + "7_block_5.jpg")
        save_feature_map(relu(res), 27, PATH_FMAPS + "7_block_5_relu.jpg")

        res = model._block_4_11[2](res)
        save_feature_map(res, 27, PATH_FMAPS + "8_block_6.jpg")
        save_feature_map(relu(res), 27, PATH_FMAPS + "8_block_6_relu.jpg")

        res = model._block_4_11[3](res)
        save_feature_map(res, 27, PATH_FMAPS + "9_block_7.jpg")
        save_feature_map(relu(res), 27, PATH_FMAPS + "9_block_7_relu.jpg")

        res = model._block_4_11[4](res)
        save_feature_map(res, 27, PATH_FMAPS + "10_block_8.jpg")
        save_feature_map(relu(res), 27, PATH_FMAPS + "10_block_8_relu.jpg")

        res = model._block_4_11[5](res)
        save_feature_map(res, 27, PATH_FMAPS + "11_block_9.jpg")
        save_feature_map(relu(res), 27, PATH_FMAPS + "11_block_9_relu.jpg")

        res = model._block_4_11[6](res)
        save_feature_map(res, 27, PATH_FMAPS + "12_block_10.jpg")
        save_feature_map(relu(res), 27, PATH_FMAPS + "12_block_10_relu.jpg")

        res = model._block_4_11[7](res)
        save_feature_map(res, 27, PATH_FMAPS + "13_block_11.jpg")
        save_feature_map(relu(res), 27, PATH_FMAPS + "13_block_11_relu.jpg")

        res = model._block_12(res)
        save_feature_map(res, 32, PATH_FMAPS + "14_block_12.jpg")
        save_feature_map(relu(res), 32, PATH_FMAPS + "14_block_12_relu.jpg")

        res = model._sep_conv_1(res)
        save_feature_map(res, 40, PATH_FMAPS + "15_sep_conv_1.jpg")

        res = model._sep_conv_2(res)
        save_feature_map(res, 46, PATH_FMAPS + "16_sep_conv_2.jpg")