import os
import re

import numpy as np
import torch


def get_best_checkpoint(path_to_checkpoint: str) -> str:
    """Checks if checkpoints exist in given folder.
    If exist the method returns best checkpoint
    built on val loss and iou.
    """
    if not os.path.exists(path_to_checkpoint):
        return None
    best_loss = float("inf")
    best_iou = 0
    best_checkpoint = None

    # Regex to match the filenames and extract loss and IOU
    pattern = re.compile(r"epoch=\d+-val_loss=([0-9.]+)-val_iou=([0-9.]+)\.ckpt")

    for filename in os.listdir(path_to_checkpoint):
        match = pattern.match(filename)
        if match:
            loss, iou = map(float, match.groups())
            # Select the checkpoint with the lowest validation loss and then the highest IOU
            if loss < best_loss or (loss == best_loss and iou > best_iou):
                best_loss, best_iou = loss, iou
                best_checkpoint = filename

    if best_checkpoint:
        return os.path.join(path_to_checkpoint, best_checkpoint)
    else:
        return None



def compute_class_weights():
    """
    Compute class weights (for DiceLoss) or alpha values (for FocalLoss) as a tensor.

    Returns:
    torch.Tensor: A tensor containing the normalized class weights or alpha values.
    """
    # Hardcoded pixel frequencies (in percentage) for each class
    pixel_frequencies = {
        'background': 57.64,
        'road': 7.24,
        'sidewalk': 25.68,
        'Bikelane': 2.18,
        'person': 0.66,
        'car': 2.94,
        'bicycle': 0.36,
        'traffic sign (front)': 0.19,
        'traffic light': 0.10,
        'Obstacle': 3.02
    }
    
    # Convert percentages to numpy array
    percentages = np.array(list(pixel_frequencies.values()))
    
    # Compute weights as inversely proportional to the percentage
    inverse_frequencies = 1.0 / percentages
    
    # Normalize weights so that they sum to 1
    normalized_weights = inverse_frequencies / np.sum(inverse_frequencies)
    
    # Convert the normalized weights to a torch tensor
    weights_tensor = torch.tensor(normalized_weights, dtype=torch.float32)
    
    return weights_tensor
