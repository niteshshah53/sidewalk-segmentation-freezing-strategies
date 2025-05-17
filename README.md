Here is a `README.md` file to help users understand how to use the sidewalk segmentation pipeline with freezing in this repository.

# sidewalk segmentation pipeline with freezing

This repository contains an sidewalk segmentation pipeline with freezing for training segmentation models. The pipeline supports various datasets and model architectures. This `README.md` provides instructions on how to set up and run using freezing techniques.

## Prerequisites

Before you begin, ensure you have met the following requirements:
* Python 3.6 or higher
* PyTorch
* Albumentations
* Pytorch Lightning
* Other dependencies listed in `requirements.txt`

## Running with Different Datasets

This code supports multiple datasets, including **Sensation** and **Cityscapes**. 

### To Use with Sensation Dataset
To run the code with the **Sensation** dataset, ensure you `training_pipeline.py` with the versions specific to Sensation.

## Installation

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd freezing
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Training with freezing enabled

To start training with freezing, use the `training_pipeline.py` script. Below are the command-line arguments you can use:

* `--ckpt`: Path where to store or load checkpoints.
* `--data_root`: Path to the dataset to use (Cityscapes or SENSATION).
* `--data_type`: Specify the dataset to be used for training (`cityscapes`, `sensation`).
* `--learning_rate`: Learning rate for fine-tuning.
* `--classes`: Number of classes to use for training.
* `--epochs`: Number of epochs to use for each active learning iteration.
* `--batch_size`: Number of batches to use during training.
* `--model_arc`: The model architecture to use during training.
* `--freeze`: Enable freeze .
* `--loss`: The loss function to use.
* `--augment`: Comma-separated list of augmentations to apply (e.g., 'rotate,flip,scale').
* `--use_class_bal`: Use class balancing for loss function.. 
* `--vis_dir`: Directory where visualization graphs are saved.
* `--freeze_strategy`: Strategy to use for layer freezing.
* `--blocks_to_freeze`: Comma-separated list of block numbers to freeze (e.g., '1,2,3'), use this argument only when --freeze_strategy is block otherwise avoid using it.
* `--cdt_freeze_compression`: Target compression rate for CDT Freeze (default = 0.5). Use only when freeze_strategy is cdt_freeze otherwise avoid using it
* `--calculate_contributions`: Calculate layer contributions and save to layer_contributions.pth. Use only when freeze_strategy is cdt_freeze.

Example command with block freezing:
python3 training_pipeline.py \
    --data_root /home/hpc/iwi5/iwi5250h/sidewalk-segmentation-pipeline/SENSATION_DS/ \
    --data_type sensation \
    --model_arc DeepLabV3Plus:efficientnet-b5:8 \
    --classes 10 \
    --batch_size 10 \
    --learning_rate 1.5e-3 \
    --epochs 500 \
    --freeze \
    --freeze_strategy block \
    --blocks_to_freeze '9, 10, 11, 12, 31, 32, 33, 34, 35' \
    --augment \
    --loss DICE \
    --ckpt /home/hpc/iwi5/iwi5250h/sidewalk-segmentation-pipeline/NewTraining/Cityscapes \
    --vis_dir /home/hpc/iwi5/iwi5250h/sidewalk-segmentation-pipeline/visualizations

Example command with contribution-driven tuning freezing:
Step 1: calculate layer contribution. To calculate layer contribution use the following command(please use this command only once, no need to calculate layer contribution again if you are using same model architecture). This will create layer_contribution.pth and will be saved in current directory.
```sh
python3 training_pipeline.py \
    --data_root /home/hpc/iwi5/iwi5250h/sidewalk-segmentation-pipeline/SENSATION_DS/ \
    --data_type sensation \
    --model_arc UnetPlusPlus:efficientnet-b0:8 \
    --classes 10 \
    --batch_size 13 \
    --learning_rate 1.5e-3 \
    --augment \
    --freeze \
    --freeze_strategy cdt_freeze \
    --calculate_contributions \
    --loss DICE \
    --ckpt /home/hpc/iwi5/iwi5250h/sidewalk-segmentation-pipeline/UnetPlusPlus/efficientnet-b0/ckpt/cityscapes \
    --vis_dir /home/hpc/iwi5/iwi5250h/sidewalk-segmentation-pipeline/visualizations

```

Step 2: Run the command normally without calculate_contributions argument. This step will use layer_contribution.pth file to freeze layers based on the contribution of layers.
```sh
python3 training_pipeline.py \
    --data_root /home/hpc/iwi5/iwi5250h/sidewalk-segmentation-pipeline/SENSATION_DS/ \
    --data_type sensation \
    --model_arc UnetPlusPlus:efficientnet-b0:8 \
    --classes 10 \
    --batch_size 13 \
    --epochs 500 \
    --learning_rate 1.5e-3 \
    --augment \
    --freeze \
    --freeze_strategy cdt_freeze \
    --loss DICE \
    --ckpt /home/hpc/iwi5/iwi5250h/sidewalk-segmentation-pipeline/UnetPlusPlus/efficientnet-b0/ckpt/cityscapes \
    --vis_dir /home/hpc/iwi5/iwi5250h/sidewalk-segmentation-pipeline/visualizations

```

### Directory Structure

* `sensation/train/builder.py`: Contains functions for creating segmentation models and preparing datasets.
* `sensation/train/data.py`: Contains dataset classes for Cityscapes and SENSATION datasets.
* `sensation/models/segmentation.py`: Contains segmentation model classes.
* `sensation/utils`: Contains utility functions for data processing, visualization, and analysis.
* `sensation/freezing`: Contains different freezing methods for freezing such as(block freezing, selective freezing, activation-based freezing, content-driven freezing, incremental defrost freezing).
* `sensation/visualization`: Contains functions to generate visualizations such as feature maps.
* `visualization.py`: Generate graphs from logs file using metrics.csv (Use this separately and only after you have log file.)