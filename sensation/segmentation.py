"""This the SENSATION segmentation class to use it in your project"""

import csv

import cv2
import numpy as np
import onnxruntime
import torch


class Segmentator:
    def __init__(
        self,
        input_width: int = 544,
        input_height: int = 544,
        model_path: str = None,
        csv_color_path: str = None,
    ):
        self.input_width = input_width
        self.input_height = input_height
        self.model_path = model_path
        if model_path is not None:
            self.onnx_session = onnxruntime.InferenceSession(model_path)
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_labels_from_csv(csv_color_path)

        # Define rgb for sidewalk in mask
        self.sidewalk_rgb = [255, 0, 0]

    def preprocess_image(self, image_array):
        image_array = image_array.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_array = (image_array - mean) / std
        return image_array


    def inference(self, images):
        size = (self.input_width, self.input_height)
        processed_images = []

        # Resize and preprocess each image in the batch
        for image in images:
            image = cv2.resize(image, size)
            image = image.astype(np.float32)
            image = self.preprocess_image(image)
            processed_images.append(image)

        # Convert the list of processed images to a numpy array and then to a tensor
        batch_images = np.array(processed_images)
        x_tensor = torch.from_numpy(batch_images).permute(0, 3, 1, 2).to(self.DEVICE)

        # Run the ONNX model for segmentation
        ort_inputs = {
        self.onnx_session.get_inputs()[0].name: x_tensor.detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    }
        ort_outputs = self.onnx_session.run(None, ort_inputs)
    
        # Process the output for each image in the batch
        predicted_outputs = [np.argmax(output, axis=0) for output in ort_outputs[0]]

        return predicted_outputs

    def mask_to_rgb(self, mask):
        """Map grayscale values in a mask to RGB values using the provided color map.

        :param mask: 2D numpy array representing a grayscale image segmentation mask

        :return: 3D numpy array representing an RGB image
        """
        rgb_image = np.zeros(
            (*mask.shape, 3), dtype=np.uint8
        )  # Initialize an RGB image with zeros

        unique_grayscales = np.unique(mask)
        for gray_value in unique_grayscales:
            if gray_value in self.color_map:
                rgb_image[mask == gray_value] = self.color_map[gray_value]

                
        return rgb_image

    def load_labels_from_csv(self, csv_color_path: str):
        """Loads the class label id and RGB values from CSV file"""
        class_label_colors = {}

        with open(csv_color_path, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                class_label = int(row["class_label"])
                red = int(row["red"])
                green = int(row["green"])
                blue = int(row["blue"])
                class_label_colors[class_label] = [red, green, blue]

            self.color_map = class_label_colors

    def get_sidewalk_rgb(self):
        return self.sidewalk_rgb


if __name__ == "__main__":
    image_path = "test/1000032755.jpg"
    save_path = "test/1000032755.png"
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    model_path = "model_weights/DeepLabV3Plus_resnet50.onnx"
    segmentator = Segmentator(model_path=model_path)
    mask = segmentator.inference(image)
    mask_rgb = segmentator.mask_to_rgb(mask)
    mask_rgb = cv2.resize(mask_rgb, (width, height))
    mask_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, mask_bgr)
