import argparse
import numpy as np
import onnxruntime as ort
import time


def load_model(model_path):
    session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    execution_providers = session.get_providers()
    print("Execution Providers:", execution_providers)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    # Assuming the input shape is [batch_size, channels, height, width]
    # This might need adjustment for models with different input formats
    channels, height, width = input_shape[1], input_shape[2], input_shape[3]
    return session, input_name, channels, height, width


def generate_random_images(height, width, channels, batch_size):
    # Adjust the data type if your model expects a different type
    return np.random.rand(batch_size, channels, height, width).astype(np.float32)


def main(model_path, num_images, batch_size):
    session, input_name, channels, height, width = load_model(model_path)

    total_time = 0.0
    durations = []
    num_batches = (num_images + batch_size - 1) // batch_size  # Ceiling division to handle any remaining images
    for i in range(num_batches):
        image_batch = generate_random_images(height, width, channels, batch_size)
        start_time = time.time()
        _ = session.run(None, {input_name: image_batch})
        duration = time.time() - start_time
        durations.append(duration)
        total_time += duration
        # print(f"Batch {i+1}/{num_batches} processing time: {duration:.4f} seconds")

    mean_duration = sum(durations) / num_batches
    fps = num_images / total_time
    print(f"Mean processing time per batch: {mean_duration:.4f} seconds")
    print(f"Total time for {num_images} images: {total_time:.4f} seconds")
    print(f"Frames per second (FPS): {fps:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimate performance of an ONNX segmentation model"
    )
    parser.add_argument(
        "model_path", type=str, help="Path to the ONNX segmentation model"
    )
    parser.add_argument(
        "num_images", type=int, help="Amount of test images to pass to the model"
    )
    parser.add_argument(
        "--batch_size", type=int, default=10, help="Batch size for inference"
    )

    args = parser.parse_args()

    main(args.model_path, args.num_images, args.batch_size)
