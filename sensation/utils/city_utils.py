import numpy as np
import cv2
import torchvision.transforms as transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2


# Image size
img_width = 1024
img_height = 512

# Define classes
valid_classes = {
    0: "unlabelled",
    7: "road",
    8: "sidewalk",        
    24: "person",
    26: "car",    
    33: "bicycle",
    20: "traffic sign",
    19: "traffic light",    
}

class_map = dict(zip(valid_classes.keys(), range(len(valid_classes))))

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

trans = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


def encode_segmap(mask, valid_classes, class_map):
    for class_idx in np.unique(mask):
        if class_idx not in valid_classes.keys():
            mask[mask == class_idx] = 0  # make the unwanted class as unlabelled

    for valid_idx in valid_classes.keys():
        mask[mask == valid_idx] = class_map[
            valid_idx
        ]  # correct the labels for valid classes

    return mask


def convert_input_images(img):
    img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
    img = trans(img)

    return img


def convert_input_masks(mask):
    # mask = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
    mask = np.array(mask)
    mask = encode_segmap(mask, valid_classes, class_map)

    # mask = torch.from_numpy(mask)
    # mask = mask.type(torch.float32)

    return mask
