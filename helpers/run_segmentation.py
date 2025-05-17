import argparse
import os
import time

import numpy as np
from PIL import Image
from tqdm import tqdm

from sensation.segmentation import Segmentator
import sensation.utils.post_process as popr

# Initialize the parser
parser = argparse.ArgumentParser(description="Segment images and save masks.")

# Add arguments
parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to the input folder containing images",
)
parser.add_argument(
    "--output",
    type=str,
    required=True,
    help="Path to the folder where output masks will be stored",
)
parser.add_argument(
    "--onnx", type=str, required=True, help="Path to the segmentation model"
)
parser.add_argument(
    "--popr",
    type=str,
    choices=["morph", "clr", "none"],
    default="none",
    help="Specify the post processing method on the output mask.",
)
parser.add_argument(
    "--batch_size", type=int, default=1, help="Batch size for segmentation"
)
# Parse the arguments
args = parser.parse_args()

# Function to load and return the segmentation model
def load_model(model_path) -> Segmentator:
    return Segmentator(model_path=model_path, input_height=640, input_width=800, csv_color_path="class_colors.csv")

# Function to perform segmentation on a batch of images
def segment_images(images, segmentator):
    return segmentator.inference(images)

# Load the segmentation model
model = load_model(args.onnx)

# Check if output folder exists, if not create it
if not os.path.exists(args.output):
    os.makedirs(args.output)

# Init all parameters
image_files = [
    f for f in os.listdir(args.input) if f.lower().endswith((".png", ".jpg", ".jpeg"))
]
start_time = time.time()
processed_images = 0
model_time = 0
batch_size = args.batch_size

for i in tqdm(range(0, len(image_files), batch_size), desc="Processing images"):
    batch_files = image_files[i:i + batch_size]
    images = []
    original_sizes = []

    for filename in batch_files:
        file_path = os.path.join(args.input, filename)
        img = Image.open(file_path)
        original_sizes.append(img.size)
        img = np.array(img)
        images.append(img)

    images = np.array(images)
    model_start_time = time.time()
    masks = segment_images(images, model)
    model_end_time = time.time()

    for j, filename in enumerate(batch_files):
        mask = masks[j]
        
        # Do post processing if needed
        if args.popr == "morph":        
            mask = popr.apply_morphological_operations(mask)
        
        mask = mask.astype(np.uint8)
        
        mask_image = Image.fromarray(mask, "L")
        # mask_image = mask_image.resize(original_sizes[j])

        output_path = os.path.join(args.output, filename)
        output_path = output_path.rsplit(".jpg", 1)[0] + ".png"

        mask_image.save(output_path)
        processed_images += 1

    model_time += (model_end_time - model_start_time)

end_time = time.time()
total_duration = end_time - start_time
mean_duration = total_duration / processed_images if processed_images > 0 else 0
fps = processed_images / model_time if model_time > 0 else 0

print(f"Processed {processed_images} images in {total_duration:.2f} seconds.")
print(f"Mean processing time per image: {mean_duration:.2f} seconds/image")
print(f"Frame rate: {fps:.2f} FPS")
print(f"Model time: {model_time:.2f} seconds")
