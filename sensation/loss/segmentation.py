import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the class frequence here
class_frequencies = [2582, 2091, 2541, 282, 608, 1194, 2051, 684, 1123, 523, 2209, 244,]
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, weights=None):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.weights = weights

    def forward(self, preds, labels):
        num_classes = preds.size(1)
        preds = preds.contiguous().view(preds.size(0), num_classes, -1)
        labels = labels.contiguous().view(labels.size(0), -1)

        preds = F.softmax(preds, dim=1)

        intersection = (preds * labels.unsqueeze(1)).sum(-1)
        preds_sum = preds.sum(-1)
        labels_sum = labels.sum(-1)

        dice = (2.0 * intersection + self.smooth) / (preds_sum + labels_sum + self.smooth)

        if self.weights is not None:
            dice = dice * self.weights

        dice_loss = 1 - dice.mean()

        return dice_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha = None, gamma:float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, labels):
        preds = preds.contiguous().view(preds.size(0), preds.size(1), -1)
        labels = labels.contiguous().view(labels.size(0), -1)

        preds = F.softmax(preds, dim=1)
        labels = F.one_hot(labels, num_classes=preds.size(1)).permute(0, 2, 1).float()

        pt = preds * labels + (1 - preds) * (1 - labels)
        log_pt = pt.log()

        if self.alpha is not None:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            log_pt = log_pt * alpha_t

        focal_loss = -((1 - pt) ** self.gamma) * log_pt
        focal_loss = focal_loss.mean()

        return focal_loss

    import torch

def compute_class_weights():
    """
    Computes class weights based on the inverse of class frequencies.

    Parameters:
    class_frequencies (list or array): List or array containing the frequency of each class.

    Returns:
    torch.Tensor: Tensor containing the normalized class weights.
    """
    inverse_class_counts = [1.0 / count for count in class_frequencies]
    total_inverse_class_counts = sum(inverse_class_counts)
    normalized_weights = [weight / total_inverse_class_counts * len(class_frequencies) for weight in inverse_class_counts]
    class_weights = torch.tensor(normalized_weights, dtype=torch.float)
    
    return class_weights
