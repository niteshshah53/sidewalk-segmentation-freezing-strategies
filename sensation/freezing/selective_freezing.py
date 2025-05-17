import pytorch_lightning as pl
import logging

logger = logging.getLogger(__name__)

def freeze_selective(model: pl.LightningModule) -> pl.LightningModule:
    """Selectively freeze layers based on predefined critical layers"""
    
    critical_layers = [
        '_expand_conv',         
        '_project_conv',        
        '_blocks.13._se_expand', 
        '_blocks.12._se_expand', 
        '_blocks.7._se_expand',   
        '_blocks.15._se_expand', 
        '_blocks.14._se_expand', 
        '_blocks.13._se_reduce', 
        '_blocks.7._se_reduce',   
        '_blocks.0',             
        '_blocks.1',
        '_blocks.2',
        '_blocks.3'             
    ]
    
    # First freeze all encoder parameters
    for name, param in model.named_parameters():
        if 'encoder' in name:
            param.requires_grad = False
            
    # Make critical layers trainable
    for name, param in model.named_parameters():
        if any(pattern in name for pattern in critical_layers):
            param.requires_grad = True
            logger.info(f"Making {name} trainable (critical layer)")
    
    # Log status
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Frozen parameters: {frozen_params}/{total_params} ({frozen_params/total_params*100:.2f}%)")
    
    return model