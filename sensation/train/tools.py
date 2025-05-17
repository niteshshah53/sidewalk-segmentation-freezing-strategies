# tools.py
import pytorch_lightning as pl
import torch
from typing import Optional, Dict, List
import logging
from sensation.freezing import freeze_by_activation, freeze_by_blocks, freeze_selective, CDTFreeze, apply_incremental_layer_defrost
from sensation.visualization import VisualizationCallback
from torch.utils.data import DataLoader  # Import DataLoader
logger = logging.getLogger(__name__)

def freeze_layers(
    model: pl.LightningModule,
    strategy: str = 'activation',
    sample_batch: Optional[torch.Tensor] = None,
    train_dataloader: Optional[DataLoader] = None,
    val_dataloader: Optional[DataLoader] = None,
    num_classes: Optional[int] = None,
    **kwargs
) -> pl.LightningModule:
    """
    Main interface for layer freezing with different strategies.
    
    Args:
        model: The PyTorch Lightning model.
        strategy: Freezing strategy ('activation', 'block', 'selective', or 'cdt_freeze').
        sample_batch: Required for activation-based freezing.
        train_dataloader: Required for incremental layer defrost.
        val_dataloader: Required for incremental layer defrost.
        num_classes: Required for incremental layer defrost.
        **kwargs: Additional arguments for specific freezing strategies.
    
    Returns:
        pl.LightningModule: The model with frozen layers.
    """
    strategies = {
        'activation': freeze_by_activation,
        'block': freeze_by_blocks,
        'selective': freeze_selective,
        'cdt_freeze': apply_cdt_freeze,
        'incremental_defrost': apply_incremental_layer_defrost  # New Incremental Layer Defrost strategy
    }
    
    if strategy not in strategies:
        raise ValueError(f"Unknown freezing strategy: {strategy}")
        
    if strategy == 'activation' and sample_batch is None:
        raise ValueError("sample_batch is required for activation-based freezing")
    
    if strategy == 'incremental_defrost' and (train_dataloader is None or val_dataloader is None or num_classes is None):
        raise ValueError("train_dataloader, val_dataloader, and num_classes are required for incremental layer defrost")
    
    logger.info(f"Applying {strategy}-based freezing strategy")
    
    # Apply freezing strategy
    if strategy == 'activation':
        model = strategies[strategy](model, sample_batch, **kwargs)
    elif strategy == 'incremental_defrost':
        model = strategies[strategy](model, train_dataloader=train_dataloader, val_dataloader=val_dataloader, num_classes=num_classes, **kwargs)
    else:
        model = strategies[strategy](model, **kwargs)
    
    # Log freezing status
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Frozen parameters: {frozen_params}/{total_params} "
               f"({frozen_params/total_params*100:.2f}%)")
    
    return model

def apply_cdt_freeze(
    model: pl.LightningModule,
    layer_contributions: Optional[Dict[str, float]] = None,
    time_compression_rate: float = 0.5,
    **kwargs
) -> pl.LightningModule:
    """
    Apply the CDT Freeze strategy to the model.
    
    Args:
        model: The PyTorch Lightning model.
        layer_contributions: A dictionary mapping layer names to their contribution scores.
        time_compression_rate: The desired time compression rate (default: 0.5).
        **kwargs: Additional arguments for CDT Freeze.
    
    Returns:
        pl.LightningModule: The model with frozen layers.
    """
    if layer_contributions is None:
        raise ValueError("layer_contributions must be provided for CDT Freeze strategy.")
    
    logger.info("Applying CDT Freeze strategy")
    
    # Initialize CDT Freeze
    cdt_freezer = CDTFreeze(model, layer_contributions, time_compression_rate)
    
    # Apply freezing
    model = cdt_freezer.apply_freezing()
    
    return model