import os
import torch
import torch.nn as nn
from typing import Dict, List
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassJaccardIndex  # IoU metric
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class IncrementalLayerDefrost:
    """
    Incremental Layer Defrosting strategy for efficient transfer learning.
    This strategy identifies the optimal number of layers to defrost based on
    the performance on the target task.
    """
    def __init__(self, model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader, num_classes: int, device: str = "cuda"):
        """
        Args:
            model (nn.Module): The model to apply the defrosting strategy to.
            train_dataloader (DataLoader): The dataloader for the training set.
            val_dataloader (DataLoader): The dataloader for the validation set.
            num_classes (int): The number of classes in the dataset.
            device (str): The device to use for computation (default: "cuda").
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.num_classes = num_classes
        self.device = device
        self.accuracy_history = []
        self.trainable_params_history = []

    def find_optimal_defrost_depth(self) -> int:
        """
        Find the optimal number of layers to defrost based on validation IoU.
        
        Returns:
            int: The optimal number of layers to defrost.
        """
        # Move the model to the specified device
        self.model = self.model.to(self.device)
        
        # Get the list of layer names
        layer_names = [name for name, _ in self.model.named_parameters()]
        
        # Initialize variables to track the best defrost depth
        best_accuracy = 0.0
        best_defrost_depth = 0
        patience = 3
        no_improvement_count = 0
        
        # Iterate over different defrost depths
        for defrost_depth in range(len(layer_names) + 1):
            # Freeze the first `defrost_depth` layers
            for idx, (_, param) in enumerate(self.model.named_parameters()):
                param.requires_grad = idx >= defrost_depth
            
            # Log the current defrost depth and parameters requiring gradients
            logger.info(f"Defrost Depth: {defrost_depth}")
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Trainable parameters: {trainable_params}/{total_params} ({trainable_params/total_params*100:.2f}%)")
            
            # Verify that there are trainable parameters
            if trainable_params == 0:
                logger.error(f"No trainable parameters found for defrost depth: {defrost_depth}")
                continue
            
            # Fine-tune the model
            accuracy = self.fine_tune_model(defrost_depth)
            logger.info(f"Defrost Depth: {defrost_depth}, Validation IoU: {accuracy:.4f}")
            
            # Update the best defrost depth if the current accuracy is better
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_defrost_depth = defrost_depth
                no_improvement_count = 0
                torch.save(self.model.state_dict(), f"best_model_defrost_depth_{defrost_depth}.pth")
            else:
                no_improvement_count += 1
            
            # Early stopping
            if no_improvement_count >= patience:
                logger.info("Early stopping triggered.")
                break
            
            # Store accuracy and trainable parameters for visualization
            self.accuracy_history.append(accuracy)
            self.trainable_params_history.append(trainable_params)
        
        return best_defrost_depth

    def fine_tune_model(self, defrost_depth: int, num_epochs: int = 1) -> float:
        """
        Fine-tune the model with the specified defrost depth.
        
        Args:
            defrost_depth (int): The number of layers to defrost.
            num_epochs (int): The number of epochs to fine-tune (default: 1).
        
        Returns:
            float: The IoU of the model after fine-tuning.
        """
        # Move the model to the specified device
        self.model = self.model.to(self.device)
        
        # Define optimizer and loss function
        trainable_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        if not trainable_params:
            logger.error("No trainable parameters found for defrost depth: {}".format(defrost_depth))
            return 0.0
        
        optimizer = torch.optim.Adam(trainable_params, lr=1e-5)  # Reduced learning rate
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)  # Handle ignored regions
        
        # Ensure the model is in training mode
        self.model.train()
        
        for epoch in range(num_epochs):
            for images, labels in self.train_dataloader:
                # Move images and labels to the specified device
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Ensure labels are of type torch.long
                labels = labels.to(torch.long)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(images)
                
                # Resize predictions to match labels
                outputs = torch.nn.functional.interpolate(
                    outputs, size=(labels.shape[1], labels.shape[2]), mode='bilinear', align_corners=False
                )
                
                # Compute loss
                loss = criterion(outputs, labels)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
        
        # Evaluate the model after fine-tuning
        accuracy = self.evaluate_model()
        return accuracy

    def evaluate_model(self) -> float:
        """
        Evaluate the model's IoU on the validation set.
        
        Returns:
            float: The IoU of the model.
        """
        self.model.eval()
        iou_metric = MulticlassJaccardIndex(num_classes=self.num_classes).to(self.device)
        total_iou = 0.0
        total_samples = 0
        with torch.no_grad():
            for images, labels in self.val_dataloader:
                # Move images and labels to the specified device
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Resize predictions to match labels
                preds = torch.nn.functional.interpolate(
                    outputs, size=(labels.shape[1], labels.shape[2]), mode='bilinear', align_corners=False
                )
                
                # Convert logits to class predictions
                preds = torch.argmax(preds, dim=1)
                
                # Compute IoU
                iou = iou_metric(preds, labels)
                total_iou += iou.item() * images.size(0)
                total_samples += images.size(0)
        
        return total_iou / total_samples

    def plot_defrosting_profile(self, output_dir: str = "./graphs/plots"):
        """
        Plot the defrosting profile with IoU and trainable parameters.
        
        Args:
            output_dir (str): Directory to save the plots.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Plot IoU vs defrost depth
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(range(len(self.accuracy_history)), self.accuracy_history, marker='o')
        plt.title('IoU vs Defrost Depth')
        plt.xlabel('Defrost Depth')
        plt.ylabel('IoU')
        plt.grid(True)
        
        # Plot trainable parameters vs defrost depth
        plt.subplot(1, 2, 2)
        plt.plot(range(len(self.trainable_params_history)), self.trainable_params_history, marker='o', color='orange')
        plt.title('Trainable Parameters vs Defrost Depth')
        plt.xlabel('Defrost Depth')
        plt.ylabel('Trainable Parameters')
        plt.grid(True)
        
        # Save the combined plot
        combined_plot_path = os.path.join(output_dir, "defrosting_profile_combined.png")
        plt.tight_layout()
        plt.savefig(combined_plot_path)
        logger.info(f"Saved combined defrosting profile plot to {combined_plot_path}")
        plt.close()

        # Plot IoU and trainable parameters on the same axes
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(self.accuracy_history)), self.accuracy_history, marker='o', label='IoU')
        plt.plot(range(len(self.trainable_params_history)), self.trainable_params_history, marker='x', label='Trainable Parameters', color='orange')
        plt.title('IoU and Trainable Parameters vs Defrost Depth')
        plt.xlabel('Defrost Depth')
        plt.ylabel('Values')
        plt.legend()
        plt.grid(True)
        
        # Save the combined plot
        combined_plot_path = os.path.join(output_dir, "defrosting_profile_combined_single_axis.png")
        plt.tight_layout()
        plt.savefig(combined_plot_path)
        logger.info(f"Saved combined defrosting profile plot on single axis to {combined_plot_path}")
        plt.close()

def apply_incremental_layer_defrost(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    num_classes: int,
    device: str = "cuda",
    **kwargs
) -> nn.Module:
    """
    Apply the Incremental Layer Defrost strategy to the model.
    
    Args:
        model (nn.Module): The model to apply the defrosting strategy to.
        train_dataloader (DataLoader): The dataloader for the training set.
        val_dataloader (DataLoader): The dataloader for the validation set.
        num_classes (int): The number of classes in the dataset.
        device (str): The device to use for computation (default: "cuda").
        **kwargs: Additional arguments for the defrosting strategy.
    
    Returns:
        nn.Module: The model with the optimal number of layers defrosted.
    """
    logger.info("Applying Incremental Layer Defrost strategy")
    
    # Initialize Incremental Layer Defrost
    defrost_strategy = IncrementalLayerDefrost(model, train_dataloader, val_dataloader, num_classes, device)
    
    # Find the optimal defrost depth
    optimal_defrost_depth = defrost_strategy.find_optimal_defrost_depth()
    logger.info(f"Optimal defrost depth: {optimal_defrost_depth}")
    
    # Defrost the optimal number of layers
    for idx, (_, param) in enumerate(model.named_parameters()):
        param.requires_grad = idx >= optimal_defrost_depth
    
    # Plot the defrosting profile
    defrost_strategy.plot_defrosting_profile(output_dir="./plots")
    
    return model