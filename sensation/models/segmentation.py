import torch
import torch.nn as nn
from torch.utils.data import Dataset
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassJaccardIndex as IoU
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau 

from sensation.models.seg_models_conf import get_seg_models_params
def create_model(model_name, model_params):
    if not hasattr(smp, model_name):
        raise ValueError(f"Model {model_name} is not a valid model name in segmentation_models_pytorch.")
    model_class = getattr(smp, model_name)
    model = model_class(**model_params)
    return model


def get_model_base_param(
    encoder_name:str = None,
    encoder_weights = None,
    in_channels = None,
        classes = None,   
                         ) -> dict:
    return {
        "encoder_name": encoder_name,
        "encoder_weights": encoder_weights,
        "in_channels": in_channels,
        "classes": classes,    
        }


class SegModel(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 8,        
        base_model = None,        
        batch_size: int = 8,
        learning_rate: float = 0.0001,
        epochs: int = 100,
        loss=None,        
        train_data: Dataset = None,
        val_data: Dataset = None,
        test_data: Dataset = None,
    ):
        super(SegModel, self).__init__()

        self.whole_model = base_model
        self.criterion = loss
        self.metrics = IoU(num_classes=num_classes)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        
        
    def forward(self, x):
        return self.whole_model(x)

    def training_step(self, batch, batch_idx):
        images, semantic_masks = batch  # float32, float32

        # Forward pass
        outputs = self.whole_model(images)  # float32
        
        loss = self.criterion(outputs, semantic_masks.long())
        iou = self.metrics(outputs, semantic_masks)

        self.log_dict({"train_loss": loss, "train_iou": iou})
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_lr)

        return {"loss": loss, "score": iou}

    def train_dataloader(self):
        return self.train_data

    def validation_step(self, batch, batch_idx):
        images, semantic_masks = batch

        # Forward pass
        outputs = self.whole_model(images)  # float32        
        # Check if output size equal input size        
        err_msg = f"Input size and output size not match - Input size:{images.size()} Output size: {outputs.size()}"
        assert (images.size(2), images.size(3)) == (outputs.size(2), outputs.size(3)), err_msg
        
        val_loss = self.criterion(outputs, semantic_masks.long())
        val_iou = self.metrics(outputs, semantic_masks)

        self.log_dict({"val_loss": val_loss, "val_iou": val_iou})

        return {"val_loss": val_loss, "val_iou": val_iou}

    def val_dataloader(self):
        return self.val_data

    def test_step(self, batch, batch_idx):
        images, semantic_masks = batch

        # Forward pass
        outputs = self.whole_model(images)  # float32

        test_loss = self.criterion(outputs, semantic_masks.long())
        test_iou = self.metrics(outputs, semantic_masks)

        self.log_dict({"test_loss": test_loss, "test_iou": test_iou})

        return {"test_loss": test_loss, "test_iou": test_iou}

    def test_dataloader(self, test_dataset=None):
        return self.test_data

    def test_epoch_end(self, outputs):
        iou_per_class = self.metrics.compute()
        self.log_dict(
            {"iou_class_" + str(i): iou.item() for i, iou in enumerate(iou_per_class)},
            prog_bar=True,
        )
        return {"iou_per_class": iou_per_class}

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.001)# weight_decay changed from 0.01
        scheduler = {
            "scheduler": OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                steps_per_epoch=int(len(self.train_data)),
                epochs=self.epochs,
                pct_start=0.3,    # from 0.3 to 0.4
                div_factor=10,    # Initial LR = max_lr/25
                final_div_factor=1000,  # Final LR = max_lr/1000
                anneal_strategy='cos'
            ),
            "interval": "step",  # or 'epoch' for epoch-wise scheduling
            "frequency": 1,
            "strict": True,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    #     scheduler = {
    #         "scheduler": ReduceLROnPlateau(
    #             optimizer,
    #             mode='max',           # Since you want to maximize IoU
    #             factor=0.5,          # Reduce LR by 20% when plateauing
    #             patience=7,          # Wait 5 epochs before reducing LR
    #             verbose=True,        # Print message when LR changes
    #             min_lr=1e-6,        # Minimum LR threshold
    #             threshold=1e-3,      # Minimum change counts as improvement
    #             cooldown=2          # Wait 2 epochs after reducing before monitoring again
    #         ),
    #         "interval": "epoch",
    #         "monitor": "val_iou",    # Monitor validation IoU
    #         "frequency": 1
    #     }
    #     return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def update_model(self, new_num_classes):
        self.metrics = IoU(num_classes=new_num_classes)

def create_base_model(model_arc:str = None,
                     in_channels:int = 3,
                     num_classes: int = None,
                     encoder_name: str = None,
                     encoder_weights:str = "imagenet",
                     ) -> nn.Module:
    
    model_params = get_model_base_param(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        )

    model_params.update(get_seg_models_params()[model_arc])
    return create_model(model_arc, model_params)