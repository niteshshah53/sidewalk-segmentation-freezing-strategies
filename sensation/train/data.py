import os

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import Cityscapes
import glob
import csv
import random


class Mapillary(Dataset):
    def __init__(self, data_path, transform, target_transform):
        self.data_path = data_path
        self.img_names = sorted(os.listdir(os.path.join(data_path, "images")))
        self.mask_names = sorted(os.listdir(os.path.join(data_path, "masks")))
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        # load image
        img_name = self.img_names[idx]
        img_path = os.path.join(self.data_path, "images", img_name)
        image = Image.open(img_path)
        image = np.array(image)

        # load mask
        mask_name = self.mask_names[idx]
        mask_path = os.path.join(self.data_path, "masks", mask_name)
        mask = Image.open(mask_path)  # read grayscale
        mask = np.array(mask)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return image, mask

    def __len__(self):
        return len(self.img_names)


class Cityscapes(Cityscapes):
    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        target_type="semantic",
        mode: str = "fine",
        transform=None,
        mask_transform=None,
    ):
        super(Cityscapes, self).__init__(
            root=dataset_path, split=split, mode=mode, target_type=target_type
        )
        self.transform = transform
        self.mask_transform = mask_transform

    def __getitem__(self, index):
        image, mask = super(Cityscapes, self).__getitem__(index)
        # Convert PIL image and mask into numpy arrays
        image = np.array(image)
        mask = np.array(mask, dtype=np.float32)
        # Apply transformations
        if self.mask_transform:
            mask = self.mask_transform(mask)
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        return image, mask


class SensationDS(Dataset):
    def __init__(
        self,
        root_dir: str,
        image_height: int = 640,
        image_width: int = 800,
        split: str = "train",
        augment: bool = False,
    ):
        self.image_height = image_height
        self.image_width = image_width
        self.split = split
        self.augment = augment

        if split == "train":
            self.root_dir = os.path.join(root_dir, "training")
        elif split == "val":
            self.root_dir = os.path.join(root_dir, "validation")
        elif split == "test":
            self.root_dir = os.path.join(root_dir, "testing")
        else:
            err_msg = f"Unsupported split: {split}. Please choose: train, val or test."
            raise ValueError(err_msg)

        images_dir = os.path.join(self.root_dir, "images")
        masks_dir = os.path.join(self.root_dir, "masks")

        self.images = sorted(
            [os.path.join(images_dir, img) for img in os.listdir(images_dir)]
        )
        self.masks = sorted(
            [os.path.join(masks_dir, mask) for mask in os.listdir(masks_dir)]
        )

        # Define transformations
        self.train_transforms = A.Compose(
            [
                A.RandomResizedCrop(
                    height=image_height,
                    width=image_width,
                    scale=(0.5, 1.0),
                    ratio=(0.45, 0.55),
                    p=1.0,
                ),  # Random crop while maintaining aspect ratio
                A.Resize(image_height, image_width),
                A.ShiftScaleRotate(
                    shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5
                ),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.1, p=0.5
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
        self.val_transform = A.Compose(
            [
                A.RandomResizedCrop(
                    height=image_height,
                    width=image_width,
                    scale=(0.5, 1.0),
                    ratio=(0.45, 0.55),
                    p=1.0,
                ),  # Random crop while maintaining aspect ratio
                A.Resize(image_height, image_width),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        # Apply transformations
        if self.split == "train" and self.augment:
            transform = self.train_transforms
        else:
            transform = self.val_transform
        augmented = transform(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]

        return image, mask


class ApolloScape(Dataset):
    def __init__(
        self,
        root_dir: str,
        csv_mapping: str,
        image_height: int = 640,
        image_width: int = 800,
        train_split: float = 0.8,
        val_split: float = 0.2,
        augment: bool = False,
    ):
        self.root_dir = root_dir
        self.image_height = image_height,
        self.image_width = image_width
        self.augment = augment
        self.split = "train"
        self.train_transforms = A.Compose(
            [
                A.Resize(image_height, image_width),
                A.ShiftScaleRotate(
                    shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5
                ),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.1, p=0.5
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
        self.val_transform = A.Compose(
            [
                A.Resize(image_height, image_width),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

        # Collect all image-mask pairs
        self.image_mask_pairs = self._collect_image_mask_pairs()

        self.train_split = train_split
        self.val_split = val_split

    def set_train_data(self):
        self.data = self.train_data
        self.split = "train"

    def set_val_data(self):
        self.data = self.val_data
        self.split = "val"

    def _collect_image_mask_pairs(self):
        image_mask_pairs = []
        color_image_dir = os.path.join(self.root_dir, "ColorImage")
        label_dir = os.path.join(self.root_dir, "masks")

        for record_dir in os.listdir(color_image_dir):
            for camera in ["Camera 5", "Camera 6"]:
                img_path_pattern = os.path.join(
                    color_image_dir, record_dir, camera, "*.jpg"
                )
                mask_path_pattern = os.path.join(label_dir, record_dir, camera, "*.png")

                img_paths = sorted(glob.glob(img_path_pattern))
                mask_paths = sorted(glob.glob(mask_path_pattern))

                # Ensure correspondence between images and masks
                for img, mask in zip(img_paths, mask_paths):
                    image_mask_pairs.append((img, mask))

        return image_mask_pairs

    def split_dataset(self):
        random.shuffle(self.image_mask_pairs)
        total_len = len(self.image_mask_pairs)
        train_len = int(total_len * self.train_split)
        val_len = int(total_len * self.val_split)

        self.train_data = self.image_mask_pairs[:train_len]
        self.val_data = self.image_mask_pairs[train_len : train_len + val_len]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, mask_path = self.data[idx]

        # Load image and mask
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))

        # Apply augmentations
        if self.split == "train" and self.augment:
            transform = self.train_transforms
        else:
            transform = self.val_transform

        augmented = transform(image=image, mask=mask)
        image, mask = augmented["image"], augmented["mask"]

        return image, mask

    def select_random(self, percentage: float = 0.8):
        total_len = len(self.image_mask_pairs)
        select_len = int(total_len * percentage)
        self.image_mask_pairs = random.sample(self.image_mask_pairs, select_len)

    def select_by_class(self, class_ids):
        selected_pairs = []
        for img_path, mask_path in self.image_mask_pairs:
            mask = np.array(Image.open(mask_path))
            mask_classes = np.unique(mask)
            if any(c in mask_classes for c in class_ids):
                selected_pairs.append((img_path, mask_path))
        self.image_mask_pairs = selected_pairs

    def info(self):
        print(f"Total images: {len(self.data)}")
        print(f"Image size: {self.image_size}")
        print(f"Classes used: {set(self.class_mapping.values()) - {0}}")
