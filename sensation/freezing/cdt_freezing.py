from collections import defaultdict
import os
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassJaccardIndex  # IoU metric
from typing import Dict, List
import torch.nn.functional as F


class CDTFreeze:
    def __init__(self, model: nn.Module, layer_contributions: Dict[str, float], time_compression_rate: float = 0.5):
        self.model = model
        self.layer_contributions = layer_contributions
        self.time_compression_rate = time_compression_rate

    def calculate_optimal_layers(self) -> List[str]:
        sorted_layers = sorted(self.layer_contributions.items(), key=lambda x: x[1], reverse=True)
        selected_layers = []
        total_cost = 0.0
        max_cost = sum(self.layer_contributions.values()) * self.time_compression_rate

        for layer_name, contribution in sorted_layers:
            if total_cost + contribution <= max_cost:
                selected_layers.append(layer_name)
                total_cost += contribution
            else:
                break

        return selected_layers

    def apply_freezing(self):
        for param in self.model.parameters():
            param.requires_grad = False

        selected_layers = self.calculate_optimal_layers()
        at_least_one_trainable = False

        for layer_name in selected_layers:
            for name, param in self.model.named_parameters():
                if layer_name in name:
                    param.requires_grad = True
                    at_least_one_trainable = True

        if not at_least_one_trainable:
            raise ValueError("No layers were set to trainable. Please check layer names.")

        self.model.apply(lambda m: m.train() if isinstance(m, nn.BatchNorm2d) else None)
        return self.model

    @staticmethod
    def evaluate_model(model, val_dataloader, num_classes, device):
        """
        Evaluate the model using IoU (Intersection over Union) metric.
        """
        model.eval()
        iou_metric = MulticlassJaccardIndex(num_classes=num_classes).to(device)  # IoU metric
        total_iou = 0.0
        total_batches = 0

        with torch.no_grad():
            for batch in val_dataloader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass to get predictions
                preds = model(inputs)

                # Resize predictions to match labels
                preds = F.interpolate(preds, size=(labels.shape[1], labels.shape[2]), mode='bilinear', align_corners=False)

                # Convert logits to class predictions
                preds = torch.argmax(preds, dim=1)

                # Compute IoU for the batch
                batch_iou = iou_metric(preds, labels)
                total_iou += batch_iou.item()
                total_batches += 1

        # Return average IoU across all batches
        return total_iou / total_batches if total_batches > 0 else 0.0

    @staticmethod
    def fine_tune_layer(model, layer_name, train_dataloader, val_dataloader, num_classes, device="cuda", num_epochs=1):
        """
        Fine-tune a specific layer and evaluate its contribution using IoU.
        """
        model = model.to(device)
        model.train()

        # Freeze all layers first
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze the specified layer
        found_layer = False
        for name, param in model.named_parameters():
            if layer_name in name:
                param.requires_grad = True
                found_layer = True

        if not found_layer:
            raise ValueError(f"Layer {layer_name} not found in model. Check layer names.")

        # Ensure BatchNorm layers remain in training mode
        model.apply(lambda m: m.train() if isinstance(m, nn.BatchNorm2d) else None)

        # Reinitialize optimizer AFTER setting requires_grad
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            for images, labels in train_dataloader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)

                # Resize predictions to match labels
                outputs = F.interpolate(outputs, size=(labels.shape[1], labels.shape[2]), mode='bilinear', align_corners=False)

                labels = labels.long()

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Evaluate the model using IoU
        return CDTFreeze.evaluate_model(model, val_dataloader, num_classes, device)

    @staticmethod
    def calculate_layer_contributions(model, train_dataloader, val_dataloader, num_classes, device="cuda"):
        """
        Calculate the contribution of each layer to the model's performance using IoU.
        """
        model = model.to(device)
        model.eval()

        # Save the original model state
        original_state_dict = model.state_dict()
        torch.save(original_state_dict, "original_model.pth")

        # Evaluate the original model
        original_iou = CDTFreeze.evaluate_model(model, val_dataloader, num_classes, device)
        layer_contributions = {}

        for name, param in model.named_parameters():
            if "weight" in name:
                print(f"Calculating contribution for layer: {name}")

                try:
                    # Fine-tune the layer and calculate new IoU
                    new_iou = CDTFreeze.fine_tune_layer(
                        model, name, train_dataloader, val_dataloader, num_classes, device
                    )
                    contribution = new_iou - original_iou
                    layer_contributions[name] = contribution
                except RuntimeError as e:
                    print(f"Skipping layer {name} due to error: {e}")

                model.load_state_dict(torch.load("original_model.pth", map_location=device))

        return layer_contributions

    @staticmethod
    def plot_layer_contributions(layer_contributions_path: str, output_path: str = "layer_contributions_plot.png"):
        """
        Generate a bar plot of layer contributions grouped by sections (encoder, decoder, segmentation head).

        Args:
            layer_contributions_path (str): Path to the layer_contributions.pth file.
            output_path (str): Path to save the generated plot.
        """
        layer_contributions = torch.load(layer_contributions_path)

        # Initialize groups
        grouped_contributions = defaultdict(float)

        for layer, contribution in layer_contributions.items():
            if "encoder" in layer:
                grouped_contributions["Encoder"] += contribution
            elif "decoder" in layer:
                grouped_contributions["Decoder"] += contribution
            elif "segmentation_head" in layer:
                grouped_contributions["Segmentation Head"] += contribution
            else:
                grouped_contributions["Other"] += contribution  # Catch-all category for unexpected layers

        # Extract grouped data
        sections = list(grouped_contributions.keys())
        contributions = list(grouped_contributions.values())

        # Plot the grouped contributions
        plt.figure(figsize=(8, 5))
        plt.barh(sections, contributions, color=['skyblue', 'lightcoral', 'lightgreen', 'gray'])
        plt.xlabel("Contribution")
        plt.ylabel("Layer Sections")
        plt.title("Grouped Layer Contributions")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()