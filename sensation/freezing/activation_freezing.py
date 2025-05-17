import os
import torch
import pytorch_lightning as pl
from typing import Dict, List, Optional
import logging
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

class ActivationAnalyzer:
    def __init__(self, model: pl.LightningModule, target_layers: List[str]):
        self.model = model
        self.target_layers = target_layers
        self.activations = {}
        self._register_hooks()

    def _register_hooks(self):
        def hook_fn(name):
            def hook(module, input, output):
                self.activations[name] = output
            return hook

        for name in self.target_layers:
            for layer_name, module in self.model.named_modules():
                if layer_name == name:
                    module.register_forward_hook(hook_fn(name))

    def analyze_batch(self, batch: torch.Tensor) -> Dict[str, Dict]:
        device = next(self.model.parameters()).device
        batch = batch.to(device)
        
        with torch.no_grad():
            _ = self.model(batch)
            
        stats = {}
        for layer_name, activation in self.activations.items():
            stats[layer_name] = {
                'mean': float(activation.mean().item()),
                'std': float(activation.std().item()),
                'max': float(activation.max().item()),
                'min': float(activation.min().item())
            }
            
        return stats

    def get_freezing_recommendation(self, stats: Dict[str, Dict]) -> Dict[str, bool]:
        recommendations = {}
        
        for layer_name, layer_stats in stats.items():
            mean_activation = layer_stats['mean']
            
            # Recommend freezing for low activation layers
            should_freeze = mean_activation < -0.05 
            recommendations[layer_name] = should_freeze
            
        return recommendations


    def plot_activation_means(self, stats: Dict[str, Dict]):
        layer_names = list(stats.keys())
        means = np.array([stats[layer]['mean'] for layer in layer_names])
        
        # Categorize activations
        colors = []
        for mean in means:
            if mean > 0.3:
                colors.append('green')  # High Activation
            elif 0.1 <= mean <= 0.3:
                colors.append('orange')  # Medium Activation
            elif 0 <= mean < 0.1:
                colors.append('blue')  # Low Activation
            else:
                colors.append('red')  # Negative Activation
        
        plt.figure(figsize=(12, 6))
        plt.bar(layer_names, means, color=colors)
        plt.xlabel("EfficientNet Blocks")
        plt.ylabel("Mean Activation Value")
        plt.title("Mean Activation per Layer with Standard Deviation")
        plt.xticks(rotation=90)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add vertical grid lines
        for i in range(len(layer_names)):
            plt.axvline(i, color='black', linestyle='-', linewidth=0.5)
        
        # Create legend
        from matplotlib.patches import Patch
        legend_patches = [
            Patch(color='green', label='High Activation (>0.3)'),
            Patch(color='orange', label='Medium Activation (0.1-0.3)'),
            Patch(color='blue', label='Low Activation (0-0.1)'),
            Patch(color='red', label='Negative Activation (<0)')
        ]
        plt.legend(handles=legend_patches)
        
        os.makedirs("visualizations", exist_ok=True)
        plt.savefig("visualizations/activation_stats.png")
        plt.show()
        
def freeze_by_activation(model: pl.LightningModule, 
                       sample_batch: torch.Tensor,
                       target_layers: Optional[List[str]] = None) -> pl.LightningModule:
    """Freeze layers based on activation analysis"""
    
    if target_layers is None:
        target_layers = [
            f'whole_model.encoder._blocks.{i}' 
            for i in range(len(model.whole_model.encoder._blocks))
        ]

    analyzer = ActivationAnalyzer(model, target_layers)
    stats = analyzer.analyze_batch(sample_batch)
    recommendations = analyzer.get_freezing_recommendation(stats)
    analyzer.plot_activation_means(stats)  # Visualization

    # Apply freezing based on recommendations
    for layer_name, should_freeze in recommendations.items():
        block_idx = int(layer_name.split('.')[-1])
        module = model.whole_model.encoder._blocks[block_idx]
        
        for param in module.parameters():
            param.requires_grad = not should_freeze
            
        status = "frozen" if should_freeze else "trainable"
        logger.info(f"Block {block_idx}: {status} (mean activation: {stats[layer_name]['mean']:.3f})")

    return model