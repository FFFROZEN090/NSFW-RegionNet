"""
Visualization utilities for keypoints, masks, and analysis results.
"""

import os
from typing import List, Optional, Tuple
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

from ..core.data_models import DetectionResult, Keypoint


class VisualizationUtils:
    """Utilities for visualizing pose detection and analysis results."""

    # COCO pose skeleton connections (bone connections)
    POSE_SKELETON = [
        [16, 14],
        [14, 12],
        [17, 15],
        [15, 13],
        [12, 13],
        [6, 12],
        [7, 13],
        [6, 7],
        [6, 8],
        [7, 9],
        [8, 10],
        [9, 11],
        [2, 3],
        [1, 2],
        [1, 3],
        [2, 4],
        [3, 5],
        [4, 6],
        [5, 7],
    ]

    # Colors for different keypoints
    KEYPOINT_COLORS = [
        (255, 0, 0),  # nose - red
        (255, 85, 0),  # left_eye - orange
        (255, 170, 0),  # right_eye - yellow-orange
        (255, 255, 0),  # left_ear - yellow
        (170, 255, 0),  # right_ear - yellow-green
        (85, 255, 0),  # left_shoulder - green
        (0, 255, 0),  # right_shoulder - bright green
        (0, 255, 85),  # left_elbow - green-cyan
        (0, 255, 170),  # right_elbow - cyan-green
        (0, 255, 255),  # left_wrist - cyan
        (0, 170, 255),  # right_wrist - cyan-blue
        (0, 85, 255),  # left_hip - blue
        (0, 0, 255),  # right_hip - bright blue
        (85, 0, 255),  # left_knee - blue-purple
        (170, 0, 255),  # right_knee - purple
        (255, 0, 255),  # left_ankle - magenta
        (255, 0, 170),  # right_ankle - pink
    ]

    @staticmethod
    def draw_keypoints(
        image: np.ndarray,
        detections: List[DetectionResult],
        draw_skeleton: bool = True,
        draw_labels: bool = True,
        keypoint_radius: int = 5,
        line_thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw keypoints and skeleton on image.

        Args:
            image: Input image (BGR format)
            detections: List of detection results containing keypoints
            draw_skeleton: Whether to draw skeleton connections
            draw_labels: Whether to draw keypoint labels
            keypoint_radius: Radius of keypoint circles
            line_thickness: Thickness of skeleton lines

        Returns:
            Image with keypoints and skeleton drawn
        """
        img_copy = image.copy()

        # Process each detection
        for detection in detections:
            # Draw skeleton connections first (so they appear under keypoints)
            if draw_skeleton:
                img_copy = VisualizationUtils._draw_skeleton(
                    img_copy, detection, line_thickness
                )

            # Draw keypoints
            for i, keypoint in enumerate(detection.keypoints):
                if keypoint.score > 0.3:  # Only draw confident keypoints
                    color = VisualizationUtils.KEYPOINT_COLORS[
                        i % len(VisualizationUtils.KEYPOINT_COLORS)
                    ]

                    # Draw keypoint circle
                    cv2.circle(
                        img_copy, (keypoint.x, keypoint.y), keypoint_radius, color, -1
                    )

                # Draw confidence circle (thicker for higher confidence)
                confidence_thickness = max(1, int(keypoint.score * 3))
                cv2.circle(
                    img_copy,
                    (keypoint.x, keypoint.y),
                    keypoint_radius + 2,
                    color,
                    confidence_thickness,
                )

                # Draw label
                if draw_labels:
                    label_text = f"{keypoint.label}:{keypoint.score:.2f}"
                    cv2.putText(
                        img_copy,
                        label_text,
                        (keypoint.x + 10, keypoint.y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1,
                    )

        return img_copy

    @staticmethod
    def _draw_skeleton(
        image: np.ndarray, detection: DetectionResult, line_thickness: int
    ) -> np.ndarray:
        """Draw skeleton connections between keypoints."""
        keypoints_dict = {kp.label: kp for kp in detection.keypoints}
        keypoint_names = [
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ]

        # Define connections (based on COCO format)
        connections = [
            ("left_shoulder", "right_shoulder"),
            ("left_shoulder", "left_elbow"),
            ("left_elbow", "left_wrist"),
            ("right_shoulder", "right_elbow"),
            ("right_elbow", "right_wrist"),
            ("left_shoulder", "left_hip"),
            ("right_shoulder", "right_hip"),
            ("left_hip", "right_hip"),
            ("left_hip", "left_knee"),
            ("left_knee", "left_ankle"),
            ("right_hip", "right_knee"),
            ("right_knee", "right_ankle"),
            ("nose", "left_eye"),
            ("nose", "right_eye"),
            ("left_eye", "left_ear"),
            ("right_eye", "right_ear"),
        ]

        for start_label, end_label in connections:
            start_kp = keypoints_dict.get(start_label)
            end_kp = keypoints_dict.get(end_label)

            if start_kp and end_kp and start_kp.score > 0.3 and end_kp.score > 0.3:
                cv2.line(
                    image,
                    (start_kp.x, start_kp.y),
                    (end_kp.x, end_kp.y),
                    (0, 255, 0),  # Green color for skeleton
                    line_thickness,
                )

        return image

    @staticmethod
    def draw_bounding_box(
        image: np.ndarray,
        detection: DetectionResult,
        color: Tuple[int, int, int] = (255, 0, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw bounding box around detection."""
        img_copy = image.copy()
        x1, y1, x2, y2 = detection.bbox

        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)

        # Add confidence label
        label = f"Person {detection.person_id}: {detection.confidence:.2f}"
        cv2.putText(
            img_copy,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            thickness,
        )

        return img_copy

    @staticmethod
    def draw_bounding_boxes(
        image: np.ndarray,
        detections: List[DetectionResult],
        color: Tuple[int, int, int] = (255, 0, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw bounding boxes around multiple detections."""
        img_copy = image.copy()

        for detection in detections:
            img_copy = VisualizationUtils.draw_bounding_box(
                img_copy, detection, color, thickness
            )

        return img_copy

    @staticmethod
    def draw_prompts(
        image: np.ndarray, points: np.ndarray, labels: np.ndarray, point_size: int = 8
    ) -> np.ndarray:
        """
        Draw SAM2 prompt points on image.

        Args:
            image: Input image
            points: Array of points (N, 2)
            labels: Array of labels (N,) - 1 for positive, 0 for negative
            point_size: Size of prompt points

        Returns:
            Image with prompt points drawn
        """
        img_copy = image.copy()

        for point, label in zip(points, labels):
            x, y = int(point[0]), int(point[1])

            if label == 1:  # Positive prompt
                color = (0, 255, 0)  # Green
                cv2.circle(img_copy, (x, y), point_size, color, -1)
                cv2.putText(
                    img_copy,
                    "+",
                    (x - 3, y + 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
            else:  # Negative prompt
                color = (0, 0, 255)  # Red
                cv2.circle(img_copy, (x, y), point_size, color, -1)
                cv2.putText(
                    img_copy,
                    "-",
                    (x - 3, y + 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

        return img_copy

    @staticmethod
    def draw_mask_overlay(
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.5,
        color: Tuple[int, int, int] = (0, 255, 255),
    ) -> np.ndarray:
        """Draw mask as colored overlay on image."""
        img_copy = image.copy()

        # Create colored overlay
        overlay = np.zeros_like(image)
        overlay[mask > 0] = color

        # Blend with original image
        result = cv2.addWeighted(img_copy, 1 - alpha, overlay, alpha, 0)

        return result

    @staticmethod
    def create_detection_summary(
        image: np.ndarray,
        detections: List[DetectionResult],
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Create a comprehensive visualization showing all detections.

        Args:
            image: Input image
            detections: List of detection results
            save_path: Optional path to save the result

        Returns:
            Combined visualization image
        """
        if not detections:
            return image.copy()

        result_img = image.copy()

        # Draw all detections
        for detection in detections:
            # Draw bounding box
            result_img = VisualizationUtils.draw_bounding_box(result_img, detection)

            # Draw keypoints and skeleton
            result_img = VisualizationUtils.draw_keypoints(
                result_img,
                [detection],
                draw_skeleton=True,
                draw_labels=False,  # Skip labels to avoid clutter
            )

        # Add summary text
        text_y = 30
        cv2.putText(
            result_img,
            f"Detected {len(detections)} person(s)",
            (10, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        if save_path:
            cv2.imwrite(save_path, result_img)
            print(f"Detection summary saved to: {save_path}")

        return result_img

    @staticmethod
    def create_pipeline_visualization(
        original_image: np.ndarray,
        detection: DetectionResult,
        prompts_points: np.ndarray,
        prompts_labels: np.ndarray,
        chest_mask: Optional[np.ndarray] = None,
        save_dir: Optional[str] = None,
    ) -> dict:
        """
        Create step-by-step visualization of the analysis pipeline.

        Args:
            original_image: Original input image
            detection: Detection result
            prompts_points: SAM2 prompt points
            prompts_labels: SAM2 prompt labels
            chest_mask: Chest triangle mask (optional)
            save_dir: Directory to save visualizations

        Returns:
            Dictionary containing all visualization images
        """
        visualizations = {}

        # 1. Original with keypoints
        keypoints_img = VisualizationUtils.draw_keypoints(original_image, [detection])
        visualizations["keypoints"] = keypoints_img

        # 2. Bounding box
        bbox_img = VisualizationUtils.draw_bounding_boxes(original_image, [detection])
        visualizations["bounding_box"] = bbox_img

        # 3. SAM2 prompts
        prompts_img = VisualizationUtils.draw_prompts(
            original_image, prompts_points, prompts_labels
        )
        visualizations["prompts"] = prompts_img

        # 4. Chest triangle mask
        if chest_mask is not None:
            chest_img = VisualizationUtils.draw_mask_overlay(original_image, chest_mask)
            visualizations["chest_triangle"] = chest_img

        # 5. Combined view
        combined_img = original_image.copy()
        combined_img = VisualizationUtils.draw_keypoints(
            combined_img, [detection], draw_labels=False
        )
        combined_img = VisualizationUtils.draw_prompts(
            combined_img, prompts_points, prompts_labels
        )
        if chest_mask is not None:
            combined_img = VisualizationUtils.draw_mask_overlay(
                combined_img, chest_mask, alpha=0.3
            )
        visualizations["combined"] = combined_img

        # Save all visualizations if directory provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            for name, img in visualizations.items():
                save_path = os.path.join(save_dir, f"{name}.png")
                cv2.imwrite(save_path, img)
                print(f"Saved {name} visualization to: {save_path}")

        return visualizations

    @staticmethod
    def create_combined_visualization(
        image: np.ndarray,
        detection: DetectionResult,
        points: np.ndarray,
        labels: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Create a combined visualization showing keypoints, prompts and segmentation mask."""
        # Start with original image
        combined_img = image.copy()

        # Draw keypoints and skeleton
        combined_img = VisualizationUtils.draw_keypoints(combined_img, [detection])

        # Draw bounding box
        combined_img = VisualizationUtils.draw_bounding_box(combined_img, detection)

        # Draw prompts
        combined_img = VisualizationUtils.draw_prompts(combined_img, points, labels)

        # Overlay segmentation mask with transparency
        if mask is not None and mask.max() > 0:
            # Create colored mask overlay
            mask_colored = np.zeros_like(image)
            mask_colored[mask > 0] = [0, 255, 0]  # Green overlay

            # Apply mask with transparency
            alpha = 0.3
            combined_img = cv2.addWeighted(
                combined_img, 1 - alpha, mask_colored, alpha, 0
            )

        return combined_img
