# %% Libraries

import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# %% Functions

def calculate_aspect_ratio(image_path: str) -> float:

    """
    Calculate the aspect ratio of an image.

    Args:
        image_path (str): Path to the image file.

    Returns:
        float: Aspect ratio of the image (width / height).
    """

    img = cv2.imread(image_path)

    if img is None:

        return None

    height, width = img.shape[:2] 

    return width / height

def resize_image(image_path: str, target_size: tuple) -> np.ndarray:

    """
    Resize an image to a target size.

    Args:
        image_path (str): Path to the image file.
        target_size (tuple): Target size (width, height) for the resized image.

    Returns:
        numpy array: Resized image.
    """

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, target_size)

    return img

def calculate_image_similarity(img1: np.ndarray, img2: np.ndarray) -> float:

    """
    Calculate the structural similarity between two images.

    Args:
        img1 (numpy array): First image.
        img2 (numpy array): Second image.

    Returns:
        float: Structural similarity between the two images.
    """

    return ssim(img1, img2, multichannel = False)

def process_image_pair(img1_path: str, img2_path: str, target_size: tuple) -> tuple:

    """
    Process a pair of images by resizing them and calculating their similarity.

    Args:
        img1_path (str): Path to the first image file.
        img2_path (str): Path to the second image file.
        target_size (tuple): Target size (width, height) for the resized images.

    Returns:
        tuple: A tuple containing the image paths and their similarity.
    """

    img1 = resize_image(img1_path, target_size)
    img2 = resize_image(img2_path, target_size)
    similarity = calculate_image_similarity(img1, img2)

    return (img1_path, img2_path), similarity

def detect_duplicates_and_similar_images(image_folder: str, threshold_duplicate: float = 0.9, threshold_similarity: float = 0.95, target_size: tuple = (128, 128)) -> tuple:

    """
    Detect duplicate and similar images in a folder.

    Args:
        image_folder (str): Path to the folder containing the images.
        threshold_duplicate (float): Threshold for duplicate images (default: 0.9).
        threshold_similarity (float): Threshold for similar images (default: 0.95).
        target_size (tuple): Target size (width, height) for the resized images (default: (128, 128)).

    Returns:
        tuple: A tuple containing two dictionaries: duplicates and similar_images.
    """

    image_files = os.listdir(image_folder)
    valid_image_files = []
    aspect_ratios = {}

    # Calculate aspect ratios for all images
    for img_file in image_files:

        img_path = os.path.join(image_folder, img_file)
        aspect_ratio = calculate_aspect_ratio(img_path)

        if aspect_ratio is not None:

            valid_image_files.append(img_file)
            aspect_ratios[img_file] = aspect_ratio

    # Process image pairs in parallel
    with ProcessPoolExecutor() as executor:

        futures = []

        for i, img1_file in enumerate(valid_image_files):

            img1_path = os.path.join(image_folder, img1_file)

            for j in range(i + 1, len(valid_image_files)):

                img2_file = valid_image_files[j]
                img2_path = os.path.join(image_folder, img2_file)

                if abs(aspect_ratios[img1_file] - aspect_ratios[img2_file]) > 0.2:

                    continue

                futures.append(executor.submit(process_image_pair, img1_path, img2_path, target_size))

        duplicates = defaultdict(list)
        similar_images = defaultdict(list)

        # Process results
        for future in concurrent.futures.as_completed(futures):

            pair, similarity = future.result()
            img1_path, img2_path = pair

            if similarity > threshold_similarity:

                similar_images[img1_path].append(img2_path)
                similar_images[img2_path].append(img1_path)

            if similarity > threshold_duplicate:

                duplicates[img1_path].append(img2_path)
                duplicates[img2_path].append(img1_path)

    return duplicates, similar_images

def remove_duplicates_and_similar_images(image_folder: str, duplicates: dict, similar_images: dict) -> None:
    
    """
    Remove duplicate and similar images from a folder.

    Args:
        image_folder (str): Path to the folder containing the images.
        duplicates (dict): Dictionary of duplicate images.
        similar_images (dict): Dictionary of similar images.
    """

    total_images = sum(len(paths) for paths in duplicates.values()) + sum(len(paths) for paths in similar_images.values())
    progress_bar = tqdm(total=total_images, desc = "Removing duplicates and similar images")

    # Remove duplicates
    for img_path, duplicate_paths in duplicates.items():

        for duplicate_path in duplicate_paths:

            if os.path.exists(duplicate_path):

                os.remove(duplicate_path)
                progress_bar.update(1)
    
    # Remove similar images
    for img_path, similar_paths in similar_images.items():

        for similar_path in similar_paths:

            if os.path.exists(similar_path) and similar_path not in duplicates:

                os.remove(similar_path)
                progress_bar.update(1)

    progress_bar.close()

def main(dataset_path: str) -> None:

    """
    Main function.
    """

    folder_name = input("Enter the name of the folder to be cleaned: ")
    output_path = dataset_path + folder_name

    duplicates, similar_images = detect_duplicates_and_similar_images(output_path)
    remove_duplicates_and_similar_images(output_path, duplicates, similar_images)

# %% Main

if __name__ == "__main__":

    dataset_path = './datasets/items/'
    main(dataset_path)