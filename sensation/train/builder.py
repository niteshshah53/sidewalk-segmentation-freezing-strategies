import copy
import logging
import os
from enum import Enum, auto

import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

from sensation.models.segmentation import SegModel, create_base_model
from sensation.train.data import ApolloScape, Cityscapes, Mapillary, SensationDS
from sensation.utils import city_utils, map_utils
from sensation.utils.data import compute_class_weights

logger = logging.getLogger(__name__)


class LossFunction(Enum):
    JACCARD = auto()
    DICE = auto()
    TVERSKY = auto()
    FOCAL = auto()
    CROSS = auto()
    LOVASZ = auto()
    SOFTBCE = auto()
    SOFTCROSS = auto()
    MCC = auto()


class LossMode(Enum):
    BINARY = auto()
    MULTICLASS = auto()
    MULTILABEL = auto()


def get_loss(
    loss_func: LossFunction = LossFunction.DICE,
    mode: LossMode = LossMode.MULTICLASS,
    use_class_bal: bool = False,
):
    loss_mode = None
    loss = None
    if use_class_bal:
        weights = compute_class_weights()
    else:
        weights = None

    if mode == LossMode.BINARY:
        loss_mode = smp.losses.constants.BINARY_MODE
    elif mode == LossMode.MULTICLASS:
        loss_mode = smp.losses.constants.MULTICLASS_MODE
    elif mode == LossMode.MULTILABEL:
        loss_mode = smp.losses.constants.MULTILABEL_MODE
    else:
        err_msg = f"Unsupported loss mode: {mode}"
        raise ValueError(err_msg)

    if loss_func == LossFunction.JACCARD:
        loss = smp.losses.JaccardLoss(mode=loss_mode)
    elif loss_func == LossFunction.DICE:
        loss = smp.losses.DiceLoss(mode=loss_mode, classes=weights)
    elif loss_func == LossFunction.SOFTCROSS:
        loss = smp.losses.SoftCrossEntropyLoss(
            reduction="mean", smooth_factor=0.4, ignore_index=-100, dim=1
        )
    elif loss_func == LossFunction.TVERSKY:
        """FP and FN is weighted by alpha and beta params. With alpha == beta == 0.5, this loss becomes equal DiceLoss."""
        loss = smp.losses.TverskyLoss(mode=loss_mode, alpha=0.7, beta=0.3, gamma=1.0)
    elif loss_func == LossFunction.CROSS:
        loss = torch.nn.CrossEntropyLoss(weight=weights)
    elif loss_func == LossFunction.FOCAL:
        loss = smp.losses.FocalLoss(
            mode=loss_mode,
            alpha=weights,
            gamma=2.0,
            ignore_index=None,
            reduction="mean",
            normalized=False,
            reduced_threshold=None,
        )
    else:
        err_msg = f"Not supported loss function: {loss_func}"
        raise ValueError(err_msg)

    return loss


def create_seg_model(
    model_arc: str = None,
    epochs: int = 1,
    num_classes: int = 8,
    learning_rate: float = 1e-3,
    batch_size: int = 1,
    ckpt_path: str = "",
    loss=None,
    train_dataloader: DataLoader = None,
    val_dataloader: DataLoader = None,
    test_dataloader: DataLoader = None,
):
    logger.info("Starting to create segmentation model.")
    base_model = None
    model_name = None
    encoder_name = None
    model_output = None
    parts = model_arc.split(":")
    if len(parts) == 3:
        model_name = parts[0]
        encoder_name = parts[1]
        model_output = int(parts[2])
        logger.debug(
            f"Detected segmentation model: {model_name} with encode: {encoder_name}."
        )
    else:
        error_msg = f"The model architecture is not correct defined in: {model_arc}"
        raise ValueError(error_msg)

    base_model = create_base_model(
        model_arc=model_name, encoder_name=encoder_name, num_classes=model_output
    )

    model = SegModel(
        num_classes=model_output,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        base_model=base_model,
        loss=loss,
        train_data=train_dataloader,
        val_data=val_dataloader,
        test_data=test_dataloader,
    )

    # Check if checkpoints exist then load
    if os.path.exists(ckpt_path):
        model = load_checkpoint(model, ckpt_path, exclude_last=False)

        # change last output layer if needed
        if num_classes > model_output:
            model.whole_model.segmentation_head = create_segmentation_head(
                model, model_name, num_classes
            )
            model.update_model(num_classes)

    return model


def create_dataloaders(train, val, test, batch_size: int):
    train_loader = torch.utils.data.DataLoader(
        dataset=train,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        persistent_workers=True,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        persistent_workers=True,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        persistent_workers=True,
    )

    return train_loader, val_loader, test_loader


def prepare_cityscapes(
    root_path: str,
    batch_size: int,
        augment: bool = False,
        image_width:int = None,
        image_height:int = None,
):
    train_dataset = Cityscapes(
        root_path,
        split="train",
        mode="fine",
        target_type="semantic",
        transform=city_utils.train_transform,
        mask_transform=city_utils.convert_input_masks,
    )

    val_dataset = Cityscapes(
        root_path,
        split="val",
        mode="fine",
        target_type="semantic",
        transform=city_utils.val_transform,
        mask_transform=city_utils.convert_input_masks,
    )

    test_dataset = Cityscapes(
        root_path,
        split="test",
        mode="fine",
        target_type="semantic",
        transform=city_utils.val_transform,
        mask_transform=city_utils.convert_input_masks,
    )

    return create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size)


def prepare_mapillary(root_path: str, batch_size: int):
    train_path = os.path.join(root_path, "training")
    val_path = os.path.join(root_path, "validation")
    test_path = os.path.join(root_path, "testing")
    train_dataset = Mapillary(
        train_path,
        transform=map_utils.train_transform,
        target_transform=map_utils.convert_input_masks,
    )
    val_dataset = Mapillary(
        val_path,
        transform=map_utils.val_transform,
        target_transform=map_utils.convert_input_masks,
    )

    test_dataset = Mapillary(
        test_path,
        transform=map_utils.val_transform,
        target_transform=map_utils.convert_input_masks,
    )

    return create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size)


def prepare_sensation(
    root_dir: str,
    batch_size: int,
    image_height: int = 640,
    image_width: int = 800,
    augment: bool = False,
):
    train_dataset = SensationDS(
        root_dir=root_dir,
        split="train",
        image_height=image_height,
        image_width=image_width,
        augment=augment,
    )

    val_dataset = SensationDS(
        root_dir=root_dir,
        split="val",
        image_height=image_height,
        image_width=image_width,
    )

    test_dataset = SensationDS(
        root_dir=root_dir,
        split="test",
        image_height=image_height,
        image_width=image_width,
    )

    return create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size)


def prepare_apolloscape(
    root_dir: str,
    batch_size: int,
    image_height: int = 640,
    image_width: int = 800,
    augment: bool = False,
):
    # Should be changed
    mapping_csv = root_dir + "/class_mapping.csv"

    train_dataset = ApolloScape(
        root_dir=root_dir,
        csv_mapping=mapping_csv,
        image_height=image_height,
        image_width=image_width,
        augment=augment,
    )

    # Shuffle dataset
    train_dataset.select_random()

    train_dataset.split_dataset()
    train_dataset.set_train_data()
    val_dataset = copy.deepcopy(train_dataset)
    val_dataset.set_val_data()

    # No test data is needed
    test_dataset = val_dataset
    return create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size)


def load_checkpoint(model, checkpoint_path, exclude_last=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])

    return model


def create_segmentation_head(model, model_name, num_classes):
    """Returns a specific segmentation head for a model."""
    if model_name in ["PSPNet", "FPN"]:
        return smp.base.SegmentationHead(
            in_channels=model.whole_model.segmentation_head[0].in_channels,
            out_channels=num_classes,
            kernel_size=model.whole_model.segmentation_head[0].kernel_size[0],
            upsampling=model.whole_model.segmentation_head[1].scale_factor,
        )
    else:
        return smp.base.SegmentationHead(
            in_channels=model.whole_model.segmentation_head[0].in_channels,
            out_channels=num_classes,
            kernel_size=3,
        )
