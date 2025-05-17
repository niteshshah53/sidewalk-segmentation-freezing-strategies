import argparse

import torch
import torch.onnx

from sensation.train import builder

# We use 8 classes of cityscapes
num_classes = 8


def export_to_onnx(pytorch_model_path: str, onnx_model_path: str, model_arc: str, batch_size: int):
    # Load the PyTorch model
    model = builder.create_seg_model(
        model_arc=model_arc,
        num_classes=1,
        ckpt_path=pytorch_model_path,
    )

    base_model = model.whole_model
    base_model.eval()

    # Example input dimensions
    input_channels = 3
    input_height = 640
    input_width = 800
    print("padding done")
    # Dummy input for tracing the model
    dummy_input = torch.randn(batch_size, input_channels, input_height, input_width)

    # Apply padding to the input
    # padded_input = F.pad(dummy_input, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

    # Export the model to ONNX format
    onnx_path = onnx_model_path
    print("Exporting the model to ONNX...")
    torch.onnx.export(
        base_model,
        dummy_input,
        onnx_path,
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
    )
    print(f"Model exported to: {onnx_path}")


def main():
    print("Starting export of onnx...")

    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX format")
    parser.add_argument(
        "--pytorch", required=True, type=str, help="Path to PyTorch model (.pt)"
    )
    parser.add_argument(
        "--onnx", required=True, type=str, help="Path to store the ONNX output"
    )
    parser.add_argument(
        "--model_arc",
        default="deeplabv3plus_resnet18",
        type=str,
        help="Define to main model architecture to use.",
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Model input batch size",
    )

    args = parser.parse_args()

    # Export to ONNX
    export_to_onnx(args.pytorch, args.onnx, args.model_arc, args.batch_size)


if __name__ == "__main__":
    main()
