"""
Segment Anything Model 2 (SAM2) wrapper for image and video segmentation.
"""

import os
import sys
from typing import Optional, Tuple
import numpy as np
import torch
import cv2

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("Warning: SAM2 not available. Please install sam2 package.")
    build_sam2 = None
    SAM2ImagePredictor = None


class SamSegmenter:
    """SAM2 segmenter for generating segmentation masks from point prompts."""

    def __init__(
        self, model_path: str, model_type: str = "hiera_large", device: str = "cuda"
    ):
        """
        Initialize SAM2 segmenter.

        Args:
            model_path: Path to SAM2 model checkpoint (.pt file)
            model_type: Model architecture type ('hiera_large', 'hiera_base_plus', 'hiera_small', 'hiera_tiny')
            device: Device to run inference on ('cpu', 'cuda', 'mps')
        """
        self.model_path = model_path
        self.model_type = model_type
        self.device = device
        self.predictor = None
        self.sam2_model = None

        # Model configuration mapping (SAM2.1 configs)
        self.model_configs = {
            "hiera_large": "sam2.1_hiera_l.yaml",
            "hiera_base_plus": "sam2.1_hiera_b+.yaml",
            "hiera_small": "sam2.1_hiera_s.yaml",
            "hiera_tiny": "sam2.1_hiera_t.yaml",
        }

        self._load_model()

    def _load_model(self) -> None:
        """Load the SAM2 model and predictor."""
        if build_sam2 is None or SAM2ImagePredictor is None:
            print("SAM2 not available. Using mock implementation for testing.")
            self.predictor = None
            return

        if not os.path.exists(self.model_path):
            print(f"Warning: SAM2 checkpoint not found at {self.model_path}")
            print(
                "Please download SAM2 checkpoint from: https://github.com/facebookresearch/segment-anything-2"
            )
            self.predictor = None
            return

        try:
            # Get config file path
            config_name = self.model_configs.get(self.model_type, "sam2.1_hiera_l.yaml")

            # Build full config path for SAM2.1
            config_path = f"configs/sam2.1/{config_name}"

            # Build SAM2 model
            self.sam2_model = build_sam2(
                config_path, self.model_path, device=self.device
            )

            # Create predictor
            self.predictor = SAM2ImagePredictor(self.sam2_model)

            print(f"SAM2 model loaded successfully on device: {self.device}")
            print(f"Model type: {self.model_type}")

        except Exception as e:
            print(f"Error loading SAM2 model: {e}")
            print("Using mock implementation for testing.")
            self.predictor = None

    def segment(
        self, image: np.ndarray, points: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """
        Generate segmentation mask from point prompts.

        Args:
            image: Input image as numpy array (H, W, C) in RGB format
            points: Point prompts as numpy array of shape (N, 2) with (x, y) coordinates
            labels: Point labels as numpy array of shape (N,) with 1 for positive, 0 for negative

        Returns:
            best_mask: Binary segmentation mask as numpy array (H, W) with boolean values
        """
        if self.predictor is None:
            # Mock implementation for testing when SAM2 is not available
            return self._generate_mock_mask(image, points, labels)

        if len(points) == 0:
            # No points provided, return empty mask
            return np.zeros((image.shape[0], image.shape[1]), dtype=bool)

        try:
            # Convert image to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Assume BGR input, convert to RGB for SAM2
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image

            # Set image for prediction
            self.predictor.set_image(image_rgb)

            # Run prediction with point prompts
            masks, scores, logits = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=True,  # Get multiple mask options
            )

            # Select best mask based on confidence score
            if len(masks) > 0:
                best_idx = np.argmax(scores)
                best_mask = masks[best_idx]

                print(
                    f"SAM2 segmentation complete. Best mask score: {scores[best_idx]:.3f}"
                )
                return best_mask.astype(bool)
            else:
                print("SAM2 did not generate any masks")
                return np.zeros((image.shape[0], image.shape[1]), dtype=bool)

        except Exception as e:
            print(f"Error during SAM2 segmentation: {e}")
            return self._generate_mock_mask(image, points, labels)

    def _generate_mock_mask(
        self, image: np.ndarray, points: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """
        Generate a mock segmentation mask for testing when SAM2 is not available.

        Args:
            image: Input image
            points: Point prompts
            labels: Point labels

        Returns:
            Mock binary mask based on positive points
        """
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)

        if len(points) == 0:
            return mask

        # Create simple circular masks around positive points
        positive_points = points[labels == 1]

        for point in positive_points:
            x, y = int(point[0]), int(point[1])

            # Create circular region around positive point
            radius = 50  # Fixed radius for mock
            y_coords, x_coords = np.ogrid[: image.shape[0], : image.shape[1]]
            distances = (x_coords - x) ** 2 + (y_coords - y) ** 2
            circular_mask = distances <= radius**2

            mask = mask | circular_mask

        print("Using mock SAM2 segmentation (circular regions around positive points)")
        return mask

    def segment_with_box(
        self, image: np.ndarray, box: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Generate segmentation mask from bounding box prompt.

        Args:
            image: Input image as numpy array (H, W, C)
            box: Bounding box as (x1, y1, x2, y2)

        Returns:
            Binary segmentation mask as numpy array (H, W)
        """
        if self.predictor is None:
            # Mock implementation
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
            x1, y1, x2, y2 = box
            mask[y1:y2, x1:x2] = True
            return mask

        try:
            # Convert image to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image

            # Set image for prediction
            self.predictor.set_image(image_rgb)

            # Run prediction with box prompt
            box_array = np.array([box])  # SAM2 expects array format
            masks, scores, logits = self.predictor.predict(
                box=box_array, multimask_output=False
            )

            if len(masks) > 0:
                return masks[0].astype(bool)
            else:
                return np.zeros((image.shape[0], image.shape[1]), dtype=bool)

        except Exception as e:
            print(f"Error during box-based segmentation: {e}")
            # Fallback to simple box mask
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
            x1, y1, x2, y2 = box
            mask[y1:y2, x1:x2] = True
            return mask

    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary containing model information
        """
        return {
            "model_path": self.model_path,
            "model_type": self.model_type,
            "device": self.device,
            "is_loaded": self.predictor is not None,
            "model_available": build_sam2 is not None
            and SAM2ImagePredictor is not None,
        }
