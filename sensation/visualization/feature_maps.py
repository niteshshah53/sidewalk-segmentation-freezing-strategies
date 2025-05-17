import torch
import pytorch_lightning as pl
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def visualize_model_features(model: pl.LightningModule,
                           sample_batch: torch.Tensor,
                           layer_name: Optional[str] = None) -> Optional[torch.Tensor]:
    device = next(model.parameters()).device
    sample_batch = sample_batch.to(device)
    
    features = {}
    def hook_fn(module, input, output):
        features['output'] = output
    
    # Print all layer names for debugging if layer_name is None
    if layer_name is None:
        layer_name = next(iter(model.named_modules()))[0]  # Default to the first named module
        logger.info("No layer specified, using first layer: %s", layer_name)
    
    target_found = False
    for name, module in model.named_modules():
        if name == layer_name:
            handle = module.register_forward_hook(hook_fn)
            target_found = True
            break
    
    if not target_found:
        logger.warning(f"Layer {layer_name} not found. Available layers:")
        for name, _ in model.named_modules():
            logger.warning(name)
        return None
        
    with torch.no_grad():
        _ = model(sample_batch)
        
    handle.remove()
    
    return features.get('output', None)
