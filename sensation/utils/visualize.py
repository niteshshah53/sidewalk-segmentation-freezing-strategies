import cv2
import numpy as np


def visualize_dominant_column(
    mask_image, target_rgb, dominant_column, line_width=3, font_size=16
):
    # Define column line color
    line_color = (255, 255, 255)

    # Create a mask for the target RGB value
    target_mask = np.all(mask_image == np.array(target_rgb)[None, None, :], axis=-1)

    # Create a new image with only the pixels defined by the target RGB value
    new_mask_np = np.zeros_like(mask_image)
    new_mask_np[target_mask] = [255, 255, 255]

    # Get the dimensions of the image
    height, width, _ = new_mask_np.shape

    # Define the column width
    column_width = width // 3

    # Draw rectangles to define the columns
    cv2.rectangle(new_mask_np, (0, 0), (width, height), line_color, line_width)
    cv2.line(
        new_mask_np,
        (column_width, 0),
        (column_width, height),
        line_color,
        thickness=line_width,
    )
    cv2.line(
        new_mask_np,
        (2 * column_width, 0),
        (2 * column_width, height),
        line_color,
        thickness=line_width,
    )

    # Put the respective text in the center of the image based on dominant_column value
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_map = {0: "No sidewalk found", 1: "Go left", 2: "Stay center", 3: "Go right"}
    text = text_map.get(dominant_column, "Invalid input")
    font_scale = font_size / 16
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, fontScale=font_scale, thickness=1
    )
    text_x = (width - text_width) // 2
    text_y = height // 5 - text_height // 2
    cv2.putText(
        new_mask_np,
        text,
        (text_x, text_y),
        font,
        font_scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return new_mask_np
