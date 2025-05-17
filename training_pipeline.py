# training_pipeline.py
import argparse
from datetime import datetime
import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger 
from sensation.freezing.cdt_freezing import CDTFreeze
from sensation.freezing import apply_incremental_layer_defrost
from sensation.visualization import VisualizationCallback
from sensation.train import builder, tools
from sensation.utils import data
import logging
import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def loss_function_arg(string):
    """Helper function to parse loss function argument"""
    try:
        return builder.LossFunction[string.upper()]
    except KeyError:
        raise argparse.ArgumentTypeError(f"{string} is not a valid loss function.")

def main():
    parser = argparse.ArgumentParser(
        description="Training pipeline for SENSATION segmentation models with visualization support."
    )
    parser.add_argument(
        "--ckpt",
        default="checkpoints",
        help="Path where to store or load checkpoints. (Default = checkpoints)",
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help="Path to the dataset to use (Cityscapes or Mapillary).",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        choices=["apolloscape", "cityscapes", "mapillary", "sensation"],
        required=True,
        help="Specify the dataset to be used for training.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-3,
        help="Learning rate (default = 5e-3)",
    )
    parser.add_argument(
        "--classes",
        type=int,
        default=10,
        help="Number of classes to use for training (default = 10)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs to use for training (default = 1).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of batches to use during training. (default = 1).",
    )
    parser.add_argument(
        "--model_arc",
        type=str,
        default="UnetPlusPlus:timm-mobilenetv3_small_075:8",
        help="The model architecture to use during training.",
    )
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="Enable freezing of layers in a model.",
        default=False,
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable augmentation of data during training with Albumentation.",
        default=False,
    )
    parser.add_argument(
        "--use_class_bal",
        action="store_true",
        help="Use class balancing for loss function.",
        default=False,
    )
    parser.add_argument(
        "--loss",
        type=loss_function_arg,
        choices=list(builder.LossFunction),
        default=builder.LossFunction.DICE,
        help="The loss function to use.",
    )
    # Added new arguments:
    parser.add_argument(
        "--vis_dir",
        type=str,
        default="visualizations",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--freeze_strategy",
        type=str,
        choices=['activation', 'block', 'selective', 'cdt_freeze', 'incremental_defrost'],
        default='selective',
        help="Strategy to use for layer freezing"
    )
    parser.add_argument(
        "--blocks_to_freeze",
        type=lambda x: [int(i) for i in x.split(',')] if x else None,
        help="Comma-separated list of block numbers to freeze (e.g., '1,2,3')"
    )
    parser.add_argument(
        "--cdt_freeze_compression",
        type=float,
        default=0.5,
        help="Target compression rate for CDT Freeze (default = 0.5)",
    )
    parser.add_argument(
        "--calculate_contributions",
        action="store_true",
        help="Calculate layer contributions and save to layer_contributions.pth.",
    )
    
    args = parser.parse_args()
    logger.info("Starting SENSATION training pipeline.")
    # Generate log file name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_file_name = f"SENSATION-Segmentation-metrics-{timestamp}"
    csv_logger = CSVLogger("logs", name=csv_file_name)
    # Prepare datasets
    logger.debug(f"Using data type: {args.data_type}.")
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
            root_dir=args.data_root,
            batch_size=args.batch_size,
            augment=args.augment,
            image_width=1024,
            image_height=512,
        )
    elif args.data_type == "apolloscape":
        train_dataset, val_dataset, _ = builder.prepare_apolloscape(
            root_dir=args.data_root,
            batch_size=args.batch_size,
            augment=args.augment,
            image_width=1024,
            image_height=512,
        )
    else:
        raise ValueError("Unknown dataset")
    logger.debug(f"Creation of data type: {args.data_type} was successful.")
    
    # Load checkpoint if exists
    ckpt_path = data.get_best_checkpoint(args.ckpt)
    if not ckpt_path:
        logger.debug("No checkpoints found. Progressing without checkpoint loading.")
        ckpt_path = ""
    else:
        logger.debug(f"Found checkpoint: {ckpt_path}.")
    # Create model
    model = builder.create_seg_model(
        model_arc=args.model_arc,
        epochs=args.epochs,
        num_classes=args.classes,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        loss=builder.get_loss(loss_func=args.loss, use_class_bal=args.use_class_bal),
        ckpt_path=ckpt_path,
        train_dataloader=train_dataset,
        val_dataloader=val_dataset,
        test_dataloader=None,
    )
    # Define callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=args.ckpt,
            filename="{epoch}-{val_loss:.5f}-{val_iou:.5f}",
            save_top_k=3,
            monitor="val_iou",
            mode="max",
        ),
        VisualizationCallback(vis_dir=args.vis_dir)
    ]
    # Freeze layers if needed
    if args.freeze:
        if args.freeze_strategy == 'activation':
            sample_batch, _ = next(iter(train_dataset))
            model = tools.freeze_layers(model, strategy='activation', sample_batch=sample_batch)
        elif args.freeze_strategy == 'block':
            model = tools.freeze_layers(model, strategy='block', blocks_to_freeze=args.blocks_to_freeze or [])
        elif args.freeze_strategy == 'selective':
            model = tools.freeze_layers(model, strategy='selective')
        elif args.freeze_strategy == 'cdt_freeze':
            if args.calculate_contributions:
                # Calculate layer contributions and save to file
                logger.info("Calculating layer contributions for CDT Freeze...")
                cdt_freezer = CDTFreeze(model, {}, args.cdt_freeze_compression)
                layer_contributions = cdt_freezer.calculate_layer_contributions(
                    model, train_dataset, val_dataset, args.classes
                )
                torch.save(layer_contributions, "layer_contributions.pth")
                logger.info("Layer contributions saved to layer_contributions.pth.")
                return  # Exit after calculating contributions
            else:
                # Generate and save the layer contributions plot
                CDTFreeze.plot_layer_contributions("layer_contributions.pth", os.path.join(args.vis_dir, "layer_contributions_plot.png"))
                logger.info("Layer contributions plot saved.")
                # Load precomputed layer contributions
                if os.path.exists("layer_contributions.pth"):
                    logger.info("Loading precomputed layer contributions for CDT Freeze...")
                    layer_contributions = torch.load("layer_contributions.pth", weights_only=True)
                else:
                    raise FileNotFoundError(
                        "layer_contributions.pth not found. Please run with --calculate_contributions first."
                    )
                # Apply CDT Freeze strategy
                cdt_freezer = CDTFreeze(model, layer_contributions, args.cdt_freeze_compression)
                model = cdt_freezer.apply_freezing()
        elif args.freeze_strategy == 'incremental_defrost':
            # Apply Incremental Layer Defrost strategy
            model = tools.freeze_layers(
                model,
                strategy='incremental_defrost',
                train_dataloader=train_dataset,
                val_dataloader=val_dataset,
                num_classes=args.classes
            )

    trainer = Trainer(
        logger=csv_logger,
        max_epochs=args.epochs,
        precision="16-mixed",
        callbacks=callbacks,
    )
    trainer.fit(model)

if __name__ == "__main__":
    main()