import argparse
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import segmentation_models_pytorch as smp
from torchmetrics.classification import MulticlassJaccardIndex as IoU

from sensation.train import builder


def loss_function_arg(string):
    try:
        return builder.LossFunction[string.upper()]
    except KeyError:
        raise argparse.ArgumentTypeError(f"{string} is not a valid loss function.")


class Discriminator(nn.Module):
    def __init__(self, input_channels=15):  # 3 channels for image + 12 channels for segmentation map
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(input_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, kernel_size=4, padding=1)
        )

    def forward(self, img, seg_map):
        # Concatenate image and segmentation map along the channel dimension
        x = torch.cat((img, seg_map), dim=1)
        return self.model(x)


class CGAN(pl.LightningModule):
    def __init__(self, generator, discriminator, learning_rate, loss):
        super(CGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.learning_rate = learning_rate
        self.adversarial_loss = nn.BCELoss()
        self.criterion = smp.losses.DiceLoss(mode='multiclass')
        self.metrics = IoU(num_classes=8)
        self.automatic_optimization = False
        

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return self.adversarial_loss(y_hat, y)


    def training_step(self, batch, batch_idx):
        real_imgs, labels = batch
        labels_one_hot = F.one_hot(labels, num_classes=12).permute(0, 3, 1, 2).float()
        valid = torch.ones(real_imgs.size(0), 1, 30, 30, device=self.device)
        fake = torch.zeros(real_imgs.size(0), 1, 30, 30, device=self.device)

        optimizer_g, optimizer_d = self.optimizers()

        # Train generator
        optimizer_g.zero_grad()
        z = torch.randn(real_imgs.size(0), 3, 256, 256, device=self.device)
        generated_masks = self.generator(z)
        g_loss = self.adversarial_loss(self.discriminator(z, generated_masks), valid)
        self.manual_backward(g_loss)
        optimizer_g.step()

        # Train discriminator
        optimizer_d.zero_grad()
        real_loss = self.adversarial_loss(self.discriminator(real_imgs, labels_one_hot), valid)
        fake_loss = self.adversarial_loss(self.discriminator(real_imgs, generated_masks.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        self.manual_backward(d_loss)
        optimizer_d.step()

        self.log('g_loss', g_loss, prog_bar=True, logger=True)
        self.log('d_loss', d_loss, prog_bar=True, logger=True)
    

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()  # Ensure labels are of type Long
            
        gen_logits = self.generator(x)

        val_loss = self.criterion(gen_logits, y)
        val_iou = self.metrics(gen_logits, y)
        self.log_dict({"val_loss": val_loss, "val_iou": val_iou}, prog_bar=True, logger=True)
        return {"val_loss": val_loss, "val_iou": val_iou}

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_g = optim.Adam(self.generator.parameters(), lr=lr)
        opt_d = optim.Adam(self.discriminator.parameters(), lr=lr)
        return [opt_g, opt_d], []

    def on_save_checkpoint(self, checkpoint):
        # Save only the generator model state dict
        checkpoint['_state_dict'] = self.generator.state_dict()
        # Remove the complete model state dict from checkpoint
#         if 'state_dict' in checkpoint:
            # del checkpoint['state_dict']
        
    def on_load_checkpoint(self, checkpoint):
        # Load the student model state dict
        self.student.load_state_dict(checkpoint['student_state_dict'])


# Main script
def main(args):
    # Data
    if args.data_type == "cityscapes":
        train_dataset, val_dataset, _ = builder.prepare_cityscapes(
            args.data_root, batch_size=args.batch_size
        )
    elif args.data_type == "mapillary":
        train_dataset, val_dataset, _ = builder.prepare_mapillary(
            args.data_root, batch_size=args.batch_size
        )
    elif args.data_type == "sensation":
        train_dataset, val_dataset, _ = builder.prepare_sensation(
            root_dir=args.data_root, batch_size=args.batch_size
        )
    else:
        raise ValueError("Unknown dataset")

    # Model
    generator = builder.create_seg_model(
        model_arc=args.model_arc,
        ckpt_path=args.ckpt,
        num_classes=args.classes,
    ).whole_model
    
    discriminator = Discriminator(input_channels=15)  # 3 channels for image + 12 channels for segmentation map

    model = CGAN(generator, discriminator, args.learning_rate, args.loss)

    ckpt_folder= os.path.dirname(args.ckpt)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_iou",
        dirpath=ckpt_folder,
        filename="student-{epoch}-{val_loss:.4f}-{val_iou:.4f}",
        save_top_k=3,
        mode="max",
    )


    # Trainer
    trainer = Trainer(max_epochs=args.epochs, callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataset, val_dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='root path to dataset')
    parser.add_argument('--data_type', type=str, required=True, help='type of dataset')
    parser.add_argument('--model_arc', type=str, required=True, help='model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to the generator checkpoint",
    )
    parser.add_argument(
        "--loss",
        type=loss_function_arg,
        choices=list(builder.LossFunction),
        default=builder.LossFunction.DICE,
        help="The loss function to use.",
    )
    parser.add_argument(
        "--classes",
        type=int,
        default=13,
        help="Number of classes in training dataset.",
    )

    args = parser.parse_args()
    main(args)
