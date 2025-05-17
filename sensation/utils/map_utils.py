import json
import torchvision.transforms as transforms
import numpy as np
import PIL

import albumentations as A
from albumentations.pytorch import ToTensorV2


# Define width and height of images divided by 16
img_width = 800
img_height = 640
trans = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

train_transform = A.Compose(
    [
        A.Resize(img_height, img_width),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


val_transform = A.Compose(
    [
        A.Resize(img_height, img_width),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


def load_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    return data


def convert_input_images(img):
    img = img.resize((img_width, img_height), resample=PIL.Image.NEAREST)
    img = trans(img)

    return img


def convert_input_masks(mask):
    # mask = mask.resize((img_width, img_height), resample=PIL.Image.NEAREST)
    # mask = np.array(mask)
    # mask = torch.from_numpy(mask)
    # mask = mask.type(torch.float32)

    return mask


def put_colors_in_grayscale_mask(grayscale_mask, n_classes, label_colors):
    # convert grayscale to color
    temp = grayscale_mask.numpy()
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, n_classes):
        r[temp == l] = label_colors[l][0]
        g[temp == l] = label_colors[l][1]
        b[temp == l] = label_colors[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b

    return rgb
