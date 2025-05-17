import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import torch.nn as nn
from torch.nn import functional as F
import logging
import seaborn as sns

logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self, model: nn.Module, target_layers: List[str]):
        self.model = model
        self.target_layers = target_layers
        self.gradients: Dict[str, torch.Tensor] = {}
        self.activations: Dict[str, torch.Tensor] = {}
        self._register_hooks()

    def _get_module_by_name(self, model, name):
        """Get a module within a model by its name string"""
        names = name.split('.')
        module = model
        for n in names:
            if hasattr(module, n):
                module = getattr(module, n)
            else:
                return None
        return module

    def _register_hooks(self):
        def save_gradient(name):
            def hook(grad):
                self.gradients[name] = grad
            return hook

        def save_activation(name):
            def hook(module, input, output):
                self.activations[name] = output
            return hook

        for name in self.target_layers:
            module = self._get_module_by_name(self.model, name)
            if module is not None:
                module.register_forward_hook(save_activation(name))
                if hasattr(module, 'register_full_backward_hook'):
                    module.register_full_backward_hook(lambda m, i, o: save_gradient(name)(o[0]))
            else:
                logger.warning(f"Module {name} not found in model")

    def get_activations(self) -> Dict[str, torch.Tensor]:
        return self.activations

    def get_gradients(self) -> Dict[str, torch.Tensor]:
        return self.gradients

def plot_activation_stats(stats: Dict[str, Dict], save_path: str):
    """Enhanced plotting of activation statistics"""
    plt.figure(figsize=(15, 8))
    
    # Prepare data for plotting
    layer_names = []
    means = []
    stds = []
    
    for layer_name, layer_stats in stats.items():
        # Convert technical names to readable format
        readable_name = layer_name.split('.')[-2:]
        readable_name = f"{readable_name[-2]}_{readable_name[-1]}"
        layer_names.append(readable_name)
        means.append(layer_stats['mean'])
        stds.append(layer_stats['std'])
    
    # Create bar plot
    bars = plt.bar(range(len(layer_names)), means, yerr=stds, capsize=5)
    
    # Color coding based on activation values
    for i, bar in enumerate(bars):
        if means[i] > 0.3:  # High activation
            bar.set_color('green')
        elif means[i] > 0.1:  # Medium activation
            bar.set_color('orange')
        elif means[i] < 0:  # Negative activation
            bar.set_color('red')
        else:  # Low activation
            bar.set_color('blue')
    
    plt.xticks(range(len(layer_names)), layer_names, rotation=45, ha='right')
    plt.xlabel('EfficientNet Blocks')
    plt.ylabel('Mean Activation Value')
    plt.title('Mean Activation per Layer with Standard Deviation')
    
    # Add grid for better readability
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='High Activation (>0.3)'),
        Patch(facecolor='orange', label='Medium Activation (0.1-0.3)'),
        Patch(facecolor='blue', label='Low Activation (0-0.1)'),
        Patch(facecolor='red', label='Negative Activation (<0)')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_feature_maps(feature_maps: torch.Tensor, 
                         save_path: str,
                         layer_name: str):
    """Enhanced visualization of feature maps"""
    num_features = min(16, feature_maps.shape[1])
    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    fig.suptitle(f'Feature Maps - {layer_name}', fontsize=16)
    
    # Create a single colorbar for all subplots
    vmin = feature_maps[:, :num_features].min()
    vmax = feature_maps[:, :num_features].max()
    
    for idx in range(num_features):
        i, j = idx // 4, idx % 4
        feature_map = feature_maps[0, idx].detach().cpu().numpy()
        im = axes[i, j].imshow(feature_map, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i, j].axis('off')
        axes[i, j].set_title(f'Channel {idx}')
    
    # Add colorbar
    plt.colorbar(im, ax=axes.ravel().tolist(), label='Activation Strength')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

class ActivationAnalyzer:
    def __init__(self, model: nn.Module, target_layers: List[str]):
        self.model = model
        self.feature_extractor = FeatureExtractor(model, target_layers)

    def analyze_batch(self, batch: torch.Tensor) -> Dict[str, Dict]:
        """Analyze activations for a batch of inputs"""
        device = next(self.model.parameters()).device
        batch = batch.to(device)
        
        with torch.no_grad():
            _ = self.model(batch)
            
        activations = self.feature_extractor.get_activations()
        
        stats = {}
        for layer_name, activation in activations.items():
            if activation is None:
                logger.warning(f"No activation found for layer {layer_name}")
                continue
                
            layer_stats = {
                'mean': float(activation.mean().item()),
                'std': float(activation.std().item()),
                'max': float(activation.max().item()),
                'min': float(activation.min().item()),
                'sparsity': float((activation == 0).float().mean().item()),
                'negative_ratio': float((activation < 0).float().mean().item())
            }
            stats[layer_name] = layer_stats
            
        return stats

    def get_freezing_recommendation(self, stats: Dict[str, Dict]) -> Dict[str, bool]:
        """Provide recommendations for layer freezing based on activation analysis"""
        recommendations = {}
        
        for layer_name, layer_stats in stats.items():
            should_freeze = True
            
            # Decision criteria
            if layer_stats['mean'] > 0.3:  # High activation
                should_freeze = True  # Freeze well-learned features
            elif layer_stats['mean'] < 0:  # Negative activation
                should_freeze = False  # Need more training
            elif layer_stats['mean'] < 0.1:  # Low activation
                should_freeze = False  # Need more training
            
            recommendations[layer_name] = should_freeze
            
        return recommendations