import pytorch_lightning as pl
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

def freeze_by_blocks(model: pl.LightningModule, blocks_to_freeze: List[int] = None) -> pl.LightningModule:
    blocks_to_freeze = blocks_to_freeze or []  # Default to empty list
    """
    Freeze specific blocks while keeping others trainable
    
    Args:
        model: The PyTorch Lightning model
        blocks_to_freeze: List of block indices to freeze
    """
    # First make all parameters trainable
    for param in model.whole_model.encoder.parameters():
        param.requires_grad = True
        
    if blocks_to_freeze:
        for i in blocks_to_freeze:
            if 0 <= i < len(model.whole_model.encoder._blocks):
                for param in model.whole_model.encoder._blocks[i].parameters():
                    param.requires_grad = False
                logger.info(f"Freezing block {i}")
            else:
                logger.warning(f"Block {i} is out of range")
    
    # Log status
    for i in range(len(model.whole_model.encoder._blocks)):
        status = "frozen" if i in (blocks_to_freeze or []) else "trainable"
        logger.info(f"Block {i} is {status}")
    
    return model