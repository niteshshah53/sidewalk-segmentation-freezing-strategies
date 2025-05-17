import os
import logging
import traceback
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback
from .feature_maps import visualize_model_features

logger = logging.getLogger(__name__)

class VisualizationCallback(Callback):
    def __init__(self, vis_dir: str):
        super().__init__()
        self.vis_dir = vis_dir
        os.makedirs(vis_dir, exist_ok=True)

    def on_train_start(self, trainer, pl_module):
        try:
            batch = next(iter(trainer.train_dataloader))
            images, _ = batch
            pl_module.train()
            self.visualize_model(pl_module, images, epoch=0)
        except Exception as e:
            logger.error(f"Error during visualization: {str(e)}")
            logger.error(traceback.format_exc())

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % 99 == 0: #generate visualization after every 99 epoch
            try:
                batch = next(iter(trainer.train_dataloader))
                images, _ = batch
                pl_module.train()
                self.visualize_model(pl_module, images, epoch=trainer.current_epoch)
            except Exception as e:
                logger.error(f"Error during visualization: {str(e)}")
                logger.error(traceback.format_exc())

    def visualize_model(self, model, images, epoch):
        epoch_dir = os.path.join(self.vis_dir, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)

        try:
            # 1. Feature Maps
            feature_maps = visualize_model_features(model, images)
            if feature_maps is not None:
                plt.figure(figsize=(15, 15))
                for idx in range(min(9, feature_maps.size(1))): #generate 16 feature map
                    plt.subplot(4, 4, idx + 1)
                    plt.imshow(feature_maps[0, idx].cpu().numpy())
                    plt.title(f'Channel {idx}')
                    plt.axis('off')
                plt.suptitle('Feature Maps', fontsize=16)
                plt.tight_layout()
                plt.savefig(os.path.join(epoch_dir, "feature_maps.png"))
                plt.close()

            # 2. Layer Status
            self._save_layer_status(model, epoch_dir)

        except Exception as e:
            logger.error(f"Error during visualization: {str(e)}")
            logger.error(traceback.format_exc())

    def _save_layer_status(self, model, save_dir):
        status_file = os.path.join(save_dir, "layer_status.txt")
        with open(status_file, 'w') as f:
            f.write("Layer Freezing Status Summary\n")
            f.write("============================\n\n")
            
            frozen_params = total_params = 0
            for name, param in model.named_parameters():
                if 'encoder' in name:
                    status = "Frozen" if not param.requires_grad else "Trainable"
                    params = param.numel()
                    frozen_params += params if not param.requires_grad else 0
                    total_params += params
                    
                    f.write(f"Layer: {name}\n")
                    f.write(f"Status: {status}\n")
                    f.write(f"Parameters: {params:,}\n")
                    f.write("-" * 50 + "\n")
            
            frozen_percentage = (frozen_params / total_params) * 100 if total_params > 0 else 0
            f.write(f"\nSummary:\n")
            f.write(f"Total Parameters: {total_params:,}\n")
            f.write(f"Frozen Parameters: {frozen_params:,} ({frozen_percentage:.2f}%)\n")
            f.write(f"Trainable Parameters: {total_params - frozen_params:,} ({100-frozen_percentage:.2f}%)\n")