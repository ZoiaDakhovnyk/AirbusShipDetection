import os
import numpy as np
from data import images_with_ships, images_without_ships


def preprocess_images(images_with_ships, images_without_ships):
    """
    Preprocess images for training a UNet model.

    Parameters:
    - images_with_ships (list): List of tuples (image, mask) where mask is a list of pixel coordinates.
    - images_without_ships (list): List of images without ships.

    Returns:
    - X (np.ndarray): Preprocessed images (concatenated ships and no-ships).
    - y (np.ndarray): Masks for ship images and no-ship images.
    """
    # Separate ships and no-ships
    ship_images = [image for image, _ in images_with_ships]
    no_ship_images = images_without_ships

    # Create masks for ship images
    ship_masks = np.zeros((len(ship_images), *ship_images[0].shape[:2]), dtype=np.uint8)
    for i, (_, mask) in enumerate(images_with_ships):
        for x, y in mask:
            if 0 <= y < ship_images[0].shape[0] and 0 <= x < ship_images[0].shape[1]:
                ship_masks[i, y, x] = 1

    # Create masks for no-ship images (all zeros)
    no_ship_masks = np.zeros((len(no_ship_images), *no_ship_images[0].shape[:2]), dtype=np.uint8)

    # Concatenate ship and no-ship images
    X = np.array(ship_images + no_ship_images)
    y = np.concatenate([ship_masks, no_ship_masks], axis=0)

    return X, y


def save_processed_data(X, y, save_dir):
    """
    Save processed data to disk.

    Parameters:
    - X (np.ndarray): Preprocessed images.
    - y (np.ndarray): Masks for ship images and no-ship images.
    - save_dir (str): Directory to save the processed data.
    """
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'X.npy'), X)
    np.save(os.path.join(save_dir, 'y.npy'), y)


if __name__ == "__main__":

    save_dir = 'C:/Users/Zoya/PycharmProjects/AirbusShipDetection/data/processed'

    # Preprocess images using data from data.py
    X, y = preprocess_images(images_with_ships, images_without_ships)

    # Save processed data
    save_processed_data(X, y, save_dir)

    print(f"Processed data saved in {save_dir}")