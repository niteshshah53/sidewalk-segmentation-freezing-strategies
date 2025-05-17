import torch
import cv2

def apply_morphological_operations(mask,  specific_classes, kernel_size=5):
    """
    Apply morphological operations to specific classes in the mask.
    
    Args:
    - mask (torch.Tensor): The predicted mask with class labels (shape [batch_size, num_classes, height, width]).
    - specific_classes (list): List of class IDs to which morphological operations should be applied.
    - kernel_size (int): Size of the morphological kernel.
    
    Returns:
    - torch.Tensor: Mask after applying morphological operations to specific classes.
    """
    # Create the kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Ensure the mask is 4D (batch_size, num_classes, height, width)
    if len(mask.shape) != 4:
        raise ValueError(f"Unexpected mask shape: {mask.shape}. Expected 4D tensor.")

    batch_size, num_classes, height, width = mask.shape

    # Create an empty array to store the processed mask (same shape as input)
    processed_mask = mask.clone()

    # Iterate over the batch
    for b in range(batch_size):
        # Iterate over all classes
        for class_id in range(num_classes):
            # Get the 2D mask for the current class
            class_mask = mask[b, class_id].cpu().numpy().astype('uint8')  # Convert to uint8 for OpenCV operations

            # Apply morphological operations only to the specific classes
            if class_id in specific_classes:
                # Apply morphological closing (dilation followed by erosion)
                closed_class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_CLOSE, kernel)

                # Apply morphological opening (erosion followed by dilation)
                refined_class_mask = cv2.morphologyEx(closed_class_mask, cv2.MORPH_OPEN, kernel)

                # Update the processed mask only for the specific class
                processed_mask[b, class_id] = torch.tensor(refined_class_mask).to(mask.device)
            else:
                # Keep the mask unchanged for unselected classes
                processed_mask[b, class_id] = mask[b, class_id]

    return processed_mask
