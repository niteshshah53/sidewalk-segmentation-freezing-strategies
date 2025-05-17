import argparse
import os
from datetime import datetime

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import MulticlassJaccardIndex as IoU

from sensation.train import builder


def loss_function_arg(string):
    try:
        return builder.LossFunction[string.upper()]
    except KeyError:
        raise argparse.ArgumentTypeError(f"{string} is not a valid loss function.")


class KnowledgeDistillationModel(pl.LightningModule):
    def __init__(self,
                 teacher,
                 student,
                 learning_rate: float = 1e-3,
                 loss=None,
                 num_classes=None,
                 temperature: float = 2.0,
                 alpha: float = 0.5,
                 beta: float = 0.5,
                 gamma: float = 0.5,
                 hard_loss_scale:float = 1.0,
                 kl_loss_scale:float = 1e-7,
                 at_loss_scale:float = 1e-5,
                 use_kl: bool = True):
        super(KnowledgeDistillationModel, self).__init__()
        self.teacher = teacher.eval()
        self.student = student
        self.learning_rate = learning_rate
        self._loss = loss
        self.metrics = IoU(num_classes=num_classes)
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.use_kl = use_kl
        self.hard_loss_scale = hard_loss_scale
        self.kl_loss_scale = kl_loss_scale
        self.at_loss_scale = at_loss_scale

        # Ensure the teacher model is in evaluation mode and gradients are disabled
        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.student(x)

    def normalize_loss(self, loss):
        mean = loss.mean()
        std = loss.std()
        return (loss - mean) / (std + 1e-6)

    def extract_features(self, model, x):
        features = []
        hooks = []

        def get_hook(features):
            def hook(module, input, output):
                features.append(output)
            return hook

        for name, layer in model.whole_model.encoder.named_children():
            hooks.append(layer.register_forward_hook(get_hook(features)))

        _ = model(x)

        for h in hooks:
            h.remove()

        return features

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()  # Ensure labels are of type Long

        with torch.no_grad():
            teacher_logits = self.teacher(x)
            teacher_features = self.extract_features(self.teacher, x)

        student_logits = self.student(x)
        student_features = self.extract_features(self.student, x)

        loss = self.compute_kd_loss(student_logits, teacher_logits, y, student_features, teacher_features)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()  # Ensure labels are of type Long

        with torch.no_grad():
            teacher_logits = self.teacher(x)
            teacher_features = self.extract_features(self.teacher, x)

        student_logits = self.student(x)
        student_features = self.extract_features(self.student, x)

        val_loss = self.compute_kd_loss(student_logits, teacher_logits, y, student_features, teacher_features)
        val_iou = self.metrics(student_logits, y)
        self.log_dict({"val_loss": val_loss, "val_iou": val_iou})
        return {"val_loss": val_loss, "val_iou": val_iou}

    def compute_kd_loss(self, student_logits, teacher_logits, labels, student_features, teacher_features):
        # Hard label loss
        ce_loss = self.hard_loss_scale * self._loss(student_logits, labels)
        
        # Soft label loss (KL Divergence)
        kl_loss = self.kl_loss_scale * F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1),
            reduction="batchmean",
        ) * (self.temperature ** 2)

        # Attention transfer loss
        if self.use_kl:
            at_loss = self.kl_divergence_attention_loss(student_features, teacher_features)
        else:
            at_loss = self.mse_attention_loss(student_features, teacher_features)

        at_loss = self.at_loss_scale * at_loss
        self.log_dict({"hard_loss": ce_loss, "kl_loss": kl_loss, "at_loss": at_loss})

        return self.alpha * ce_loss + self.beta * kl_loss + self.gamma * at_loss

    def attention_map(self, feature_map):
        return torch.sum(torch.abs(feature_map), dim=1, keepdim=True)

    def kl_divergence_attention_loss(self, student_features, teacher_features):
        kl_loss = 0.0
        for sf, tf in zip(student_features, teacher_features):
            # Compute the attention maps
            smap = self.attention_map(sf)
            tmap = self.attention_map(tf)
            # Resize student attention map to match teacher's attention map size
            smap = F.interpolate(smap, size=tmap.shape[2:], mode='bilinear', align_corners=False)
            # Normalize attention maps to make them distributions
            smap = F.softmax(smap.view(smap.size(0), -1), dim=1)
            tmap = F.softmax(tmap.view(tmap.size(0), -1), dim=1)
            # Compute the KL Divergence
            kl_loss += F.kl_div(smap.log(), tmap, reduction='batchmean')
        return kl_loss

    def mse_attention_loss(self, student_features, teacher_features):
        mse_loss = 0.0
        for sf, tf in zip(student_features, teacher_features):
            # Compute the attention maps
            smap = self.attention_map(sf)
            tmap = self.attention_map(tf)
            # Resize student attention map to match teacher's attention map size
            smap = F.interpolate(smap, size=tmap.shape[2:], mode='bilinear', align_corners=False)
            # Compute the MSE
            mse_loss += F.mse_loss(smap, tmap)
        return mse_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.student.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def on_save_checkpoint(self, checkpoint):
        # Save only the student model state dict
        checkpoint['student_state_dict'] = self.student.state_dict()
        # Remove the complete model state dict from checkpoint
        if 'state_dict' in checkpoint:
            del checkpoint['state_dict']

    def on_load_checkpoint(self, checkpoint):
        # Load the student model state dict
        self.student.load_state_dict(checkpoint['student_state_dict'])


def main(args):
    # Load your dataset
    train_dataset, val_dataset, _ = builder.prepare_sensation(
        root_dir=args.data_root, batch_size=args.batch_size
    )

    # Load teacher and student models
    teacher_model = load_model(args.teach_model_arc, args.teacher_checkpoint)
    student_model = load_model(args.stud_model_arc, args.student_checkpoint, args.stud_classes)

    model = KnowledgeDistillationModel(teacher=teacher_model,
                                       student=student_model,
                                       loss=builder.get_loss(args.loss),
                                       learning_rate=args.learning_rate,
                                       num_classes=args.classes,
                                       alpha=args.alpha,
                                       beta=args.beta,
                                       gamma=args.gamma,
                                       temperature=args.temperature,
                                       use_kl=args.use_kl)

    # Callbacks and Trainer
    stud_folder_name = os.path.dirname(args.student_checkpoint)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_iou",
        dirpath=stud_folder_name,
        filename="student-{epoch}-{val_loss:.4f}-{val_iou:.4f}",
        save_top_k=3,
        mode="max",
    )

    # Generate log file name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_file_name = f"kd_metrics_{timestamp}"
    csv_logger = pl.loggers.CSVLogger("logs", name=csv_file_name)
    trainer = Trainer(max_epochs=args.epochs, callbacks=[checkpoint_callback], logger=csv_logger)
    trainer.fit(model, train_dataset, val_dataset)


def load_model(model_arc, ckpt_path, classes: int = 1):
    return builder.create_seg_model(
        model_arc=model_arc,
        ckpt_path=ckpt_path,
        num_classes=classes,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train student model with knowledge distillation"
    )
    parser.add_argument(
        "--data_root", type=str, required=True, help="Root directory of the dataset"
    )
    parser.add_argument(
        "--teacher_checkpoint",
        type=str,
        required=True,
        help="Path to the teacher model checkpoint",
    )
    parser.add_argument(
        "--student_checkpoint",
        type=str,
        required=True,
        help="Path to the student model checkpoint",
    )
    parser.add_argument(
        "--epochs", type=int, required=True, help="Number of epochs to train"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-3, help="Learning rate for training"
    )
    parser.add_argument(
        "--teach_model_arc",
        type=str,
        required=True,
        help="Architecture of the teacher model",
    )
    parser.add_argument(
        "--stud_model_arc",
        type=str,
        required=True,
        help="Architecture of the student model",
    )
    parser.add_argument(
        "--stud_classes",
        type=int,
        help="Output of the segmentation head of the student model.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight to control the hard loss during training.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="Weight to control the KL divergence loss during training.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="Weight to control the attention transfer loss during training.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=2.0,
        help="Temperature for the KL divergence loss.",
    )
    parser.add_argument(
        "--classes",
        type=int,
        default=13,
        help="Number of classes in training dataset.",
    )    
    parser.add_argument(
        "--loss",
        type=loss_function_arg,
        choices=list(builder.LossFunction),
        default=builder.LossFunction.DICE,
        help="The loss function to use.",
    )
    parser.add_argument(
        "--use_kl",
        action='store_true',
        help="Use KL divergence for attention transfer loss instead of MSE.",
    )

    args = parser.parse_args()
    main(args)
