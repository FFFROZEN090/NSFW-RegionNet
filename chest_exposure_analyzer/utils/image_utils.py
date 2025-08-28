"""
Image loading, saving, and preprocessing utilities.
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class ImageUtils:
    """Utility class for image operations."""

    @staticmethod
    def load_image(image_path: str) -> Optional[np.ndarray]:
        """Load image from file path."""
        try:
            image = cv2.imread(image_path)
            return image
        except Exception:
            return None

    @staticmethod
    def save_image(image: np.ndarray, output_path: str) -> bool:
        """Save image to file path."""
        try:
            cv2.imwrite(output_path, image)
            return True
        except Exception:
            return False

    @staticmethod
    def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize image to target size."""
        return cv2.resize(image, target_size)

    @staticmethod
    def get_image_info(image: np.ndarray) -> dict:
        """Get image information."""
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) > 2 else 1

        return {
            "width": width,
            "height": height,
            "channels": channels,
            "shape": image.shape,
        }
