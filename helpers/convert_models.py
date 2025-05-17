import argparse
import os
import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class BaseModel(nn.Module):
    def __init__(self, seg_model=None):
        super(BaseModel, self).__init__()
        self.arc = seg_model

    def forward(self, images):
        logits = self.arc(images)
        return logits

def create_base_model(model_arc, in_channels=3, num_classes=None, encoder_name=None, encoder_weights="imagenet"):
    all_attrs = dir(smp)
    model_types = [attr for attr in all_attrs if callable(getattr(smp, attr)) and not attr.startswith("__")]

    if model_arc not in model_types:
        raise ValueError(f"Your given model:{model_arc} is not in supported list of models:{model_types}.")

    smp_model = getattr(smp, model_arc)(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
    )

    return BaseModel(seg_model=smp_model)

class SegModel(pl.LightningModule):
    def __init__(self, num_classes=8, base_model=None, batch_size=8, learning_rate=0.0001, epochs=100, loss=None, train_data=None, val_data=None, test_data=None):
        super(SegModel, self).__init__()
        self.whole_model = base_model
        self.criterion = loss
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    def forward(self, x):
        return self.whole_model(x)

    def training_step(self, batch, batch_idx):
        images, semantic_masks = batch
        outputs = self(images)
        loss = self.criterion(outputs, semantic_masks.long())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, semantic_masks = batch
        outputs = self(images)
        val_loss = self.criterion(outputs, semantic_masks.long())
        self.log('val_loss', val_loss)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

class DummyDataset(Dataset):
    def __init__(self, num_samples, image_shape, num_classes):
        self.num_samples = num_samples
        self.image_shape = image_shape
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.randn(*self.image_shape)
        mask = torch.randint(0, self.num_classes, self.image_shape[1:])
        return image, mask

def main():
    parser = argparse.ArgumentParser(description="Train and manage segmentation models.")
    parser.add_argument('--model_arc', type=str, required=True, help="Model architecture in format ModelName:EncoderName:OutputChannels")
    parser.add_argument('--checkpoint_path', type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument('--save_folder', type=str, required=True, help="Folder to save the new checkpoint")

    args = parser.parse_args()

    model_arc, encoder_name, num_classes = args.model_arc.split(':')
    num_classes = int(num_classes)

    # Initialize the loss function
    loss_fn = smp.losses.DiceLoss(mode='multiclass')

    # Create base model
    base_model = create_base_model(model_arc=model_arc, encoder_name=encoder_name, num_classes=num_classes)
    
    # Initialize Lightning model
    lightning_model = SegModel(num_classes=num_classes, base_model=base_model, loss=loss_fn)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=lambda storage, loc: storage)
    lightning_model.load_state_dict(checkpoint['state_dict'])

    # Create a new Lightning model and assign the smp model
    new_model = SegModel(num_classes=num_classes, base_model=None, loss=loss_fn)
    new_model.model= lightning_model.whole_model.arc

    # Initialize Trainer
    trainer = pl.Trainer(max_epochs=1)

    # Create dummy data loader
    dummy_data = DummyDataset(num_samples=1, image_shape=(3, 640, 800), num_classes=num_classes)
    dummy_loader = DataLoader(dummy_data, batch_size=1)

    # Validate to attach the model to the trainer
    trainer.validate(new_model, dummy_loader)

    # Save checkpoint
    save_path = os.path.join(args.save_folder, os.path.basename(args.checkpoint_path))
    trainer.save_checkpoint(save_path)

    print(f"New checkpoint saved at {save_path}")

if __name__ == "__main__":
    main()
