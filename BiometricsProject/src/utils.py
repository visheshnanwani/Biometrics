import os
import cv2
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def create_sample_dataset():
    """Create sample directory structure for testing"""
    directories = [
        'data/references',
        'data/test/genuine',
        'data/test/forged',
        'data/samples',
        'results'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

    print("\nSample dataset structure created!")
    print("Place your reference signatures in: data/references/")
    print("Place test signatures in: data/test/genuine/ or data/test/forged/")


def load_images_from_folder(folder_path):
    """Load all images from a folder"""
    image_paths = []
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

    folder = Path(folder_path)
    if not folder.exists():
        logger.warning(f"Folder {folder_path} does not exist")
        return []

    for ext in valid_extensions:
        image_paths.extend(folder.glob(f'*{ext}'))
        image_paths.extend(folder.glob(f'*{ext.upper()}'))

    return [str(path) for path in image_paths]


def resize_image(image, max_width=800, max_height=600):
    """Resize image while maintaining aspect ratio"""
    h, w = image.shape[:2]

    if w > max_width or h > max_height:
        scale = min(max_width / w, max_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h))

    return image


def validate_image(image_path):
    """Validate if image can be processed"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False, "Could not read image"

        if img.shape[0] < 10 or img.shape[1] < 10:
            return False, "Image too small"

        return True, "Valid"
    except Exception as e:
        return False, str(e)