import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def load_images_and_masks(image_dir, csv_file, resize=None):
    """
    Load images and corresponding masks from the directory and CSV file.

    Parameters:
    - image_dir (str): Directory path containing images.
    - csv_file (str): Path to CSV file containing ImageId and EncodedPixels.
    - resize (tuple): Optional, resize dimensions (width, height).

    Returns:
    - images_with_ships (list): List of tuples (image, mask) where mask is a list of pixel coordinates.
    - images_without_ships (list): List of images without ships.
    """
    df = pd.read_csv(csv_file)

    images_with_ships = []
    images_without_ships = []

    for index, row in df.iterrows():
        image_id = row['ImageId']
        image_path = os.path.join(image_dir, image_id)

        try:
            if resize:
                image = np.array(Image.open(image_path).resize(resize))
            else:
                image = np.array(Image.open(image_path))

            if pd.notna(row['EncodedPixels']):
                mask_pixels = rle_to_pixels(row['EncodedPixels'], resize=resize)
                images_with_ships.append((image, mask_pixels))
            else:
                images_without_ships.append(image)
        except OSError as e:
            print(f"Error loading image {image_path}: {e}")
            continue  # Skip this image if there's an error

    return images_with_ships, images_without_ships


def rle_to_pixels(rle_code, original_size=(768, 768), resize=None):
    """
    Transform a RLE code string into a list of pixels on a resized canvas.

    Parameters:
    - rle_code (str): RLE code string from the CSV file.
    - original_size (tuple): Original image size (width, height).
    - resize (tuple): Optional, resize dimensions (width, height).

    Returns:
    - pixels (list): List of pixel coordinates [(x1, y1), (x2, y2), ...].
    """
    original_width, original_height = original_size
    if resize:
        width, height = resize
    else:
        width, height = original_size

    rle_code = [int(i) for i in rle_code.split()]
    pixels = [(pixel_position % original_width, pixel_position // original_width)
              for start, length in list(zip(rle_code[0:-1:2], rle_code[1::2]))
              for pixel_position in range(start, start + length)]

    if resize:
        scale_x = width / original_width
        scale_y = height / original_height
        pixels = [(int(x * scale_x), int(y * scale_y)) for x, y in pixels]

    return pixels


def apply_mask(image, mask):
    """
    Apply mask to the image.

    Parameters:
    - image (np.ndarray): Input image as a numpy array.
    - mask (list): List of pixel coordinates to apply mask.

    Returns:
    - image (np.ndarray): Image with applied mask.
    """
    for x, y in mask:
        image[x, y, [0, 1]] = 255  # Apply mask to the red and green channels
    return image


def split_data(X, y, validation_split=0.2, test_split=0.1):
    # Create a binary label for stratification (1 if image contains ships, 0 otherwise)
    y_binary = (y.sum(axis=(1, 2)) > 0).astype(int)

    # Split data into train and test
    X_train, X_test, y_train, y_test, y_train_binary, y_test_binary = train_test_split(
        X, y, y_binary, stratify=y_binary, test_size=test_split, random_state=42
    )

    # Split train further into train and validation
    X_train, X_valid, y_train, y_valid, y_train_binary, y_valid_binary = train_test_split(
        X_train, y_train, y_train_binary, stratify=y_train_binary, test_size=validation_split, random_state=42
    )

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)


def sample_ships(in_df, base_rep_val=9000):
    if in_df['ships'].values[0] == 0:
        return in_df.sample(base_rep_val // 3)
    else:
        return in_df.sample(base_rep_val, replace=(in_df.shape[0] < base_rep_val))


def balance_dataset(X, y, base_rep_val=9000):
    """
    Balance the dataset based on the number of ships.

    Parameters:
    - X (np.ndarray): Preprocessed images.
    - y (np.ndarray): Masks for ship images and no-ship images.
    - base_rep_val (int): Base number of samples to replicate for balancing.

    Returns:
    - X_balanced (np.ndarray): Balanced images.
    - y_balanced (np.ndarray): Balanced masks.
    """
    ship_counts = np.sum(y, axis=(1, 2))  # Sum of masks to count ships per image
    ship_counts_df = pd.DataFrame({'ships': ship_counts})

    ship_counts_df['grouped_ship_count'] = ship_counts_df['ships'].map(lambda x: (x + 1) // 2).clip(0, 7)
    ship_counts_df['index'] = ship_counts_df.index

    balanced_indices = ship_counts_df.groupby('grouped_ship_count').apply(sample_ships, base_rep_val=base_rep_val)[
        'index'].values

    X_balanced = X[balanced_indices]
    y_balanced = y[balanced_indices]

    return X_balanced, y_balanced


if __name__ == "__main__":
    # Example usage:
    img_dir = 'C:/Users/Zoya/PycharmProjects/AirbusShipDetection/data/train'
    csv_file = 'C:/Users/Zoya/PycharmProjects/AirbusShipDetection/data/train_ship_segmentations_v2.csv'

    # Load images and masks
    images_with_ships, images_without_ships = load_images_and_masks(img_dir, csv_file, resize=(64, 64))

    # Visualize the first image with mask
    if images_with_ships:
        first_image, first_mask = images_with_ships[0]
        masked_image = apply_mask(first_image.copy(), first_mask)
        plt.imshow(masked_image)
        plt.axis('off')
        plt.show()
    else:
        print("No images with ships found in the dataset.")