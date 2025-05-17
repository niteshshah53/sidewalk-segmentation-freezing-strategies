import argparse
import math

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sensation.train import builder
import sensation.utils.post_process as popr


def print_iou_summary(ious):
    # Load class names from the CSV file
    class_data = pd.read_csv("class_colors.csv")

    # Print a header
    print("IoU Summary:")
    print("-" * 40)

    # Iterate over the IoU list and corresponding rows in the DataFrame
    for i, iou in enumerate(ious):
        class_name = class_data.loc[i, "Class_Names"]
        print(f"Class {i} - {class_name}: IoU = {iou:.4f}")


def calculate_iou(preds, labels, num_classes):
    ious = []
    preds = torch.argmax(
        preds, dim=1
    )  # Assuming model output is 'C,H,W' and C is num_classes
    for cls in range(num_classes):
        pred_cls = preds == cls
        label_cls = labels == cls
        intersection = (pred_cls & label_cls).sum().float()
        union = (pred_cls | label_cls).sum().float()

        if union == 0:
            iou = torch.tensor(float("nan"))  # Use a tensor for NaN
        else:
            iou = intersection / union

        ious.append(iou.item())  # Safe to call .item() as iou is always a tensor

    return ious


def evaluate_model(test_dataset, model, num_classes):
    class_ious = []
    for _ in range(num_classes):
        class_ious.append([])

    for images, masks in tqdm(
        test_dataset, total=len(test_dataset), desc="Evaluating model"
    ):
        if torch.cuda.is_available():
            images, masks = images.cuda(), masks.cuda()
        with torch.no_grad():
            outputs = model(images)

            if args.popr == "morph":
                outputs = popr.apply_morphological_operations(outputs)
                outputs = outputs.cuda()
                
        ious = calculate_iou(outputs, masks, num_classes)
        for i, val in enumerate(ious):
            if not math.isnan(
                val
            ):  # Using Python's math.isnan to handle float NaN checks
                class_ious[i].append(val)

    # Compute mean IoU per class avoiding NaN values
    mean_ious = [np.nanmean(iou) if iou else float("nan") for iou in class_ious]
    return mean_ious


def main(args):
    # Load the datasets
    _, val_dataloader, _ = builder.prepare_sensation(
        args.data_root, batch_size=args.batch_size
    )

    # Create the model
    model = builder.create_seg_model(
        model_arc=args.model_arc,
        num_classes=args.classes,
        batch_size=args.batch_size,
        ckpt_path=args.ckpt,
    )

    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    # Evaluate the model
    ious = evaluate_model(val_dataloader, model, args.classes)
    print_iou_summary(ious)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a Segmentation Model on SENSATION DS"
    )
    parser.add_argument(
        "--data_root", type=str, required=True, help="Path to the dataset root."
    )
    parser.add_argument(
        "--model_arc", type=str, required=True, help="Model architecture."
    )
    parser.add_argument("--classes", type=int, required=True, help="Number of classes.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Path to the checkpoint folder."
    )
    parser.add_argument(
    "--popr",
        type=str,
        choices=["morph", "clr", "none"],
        default="none",
        help="Specify the post processing method on the output mask.",
    )
    args = parser.parse_args()

    main(args)
