"""
YOLOv11-Pose model wrapper for pose detection.
"""

import os
from typing import List, Optional, Tuple
import numpy as np
import torch
from ultralytics import YOLO
import cv2

from ..data_models import DetectionResult, Keypoint


class YoloDetector:
    """YOLOv11-Pose detector for human pose estimation."""

    # COCO pose keypoint names (17 keypoints)
    KEYPOINT_NAMES = [
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

    def __init__(
        self, model_path: str, device: str = "cpu", confidence_threshold: float = 0.5
    ):
        """
        Initialize YOLOv11-Pose detector.

        Args:
            model_path: Path to YOLOv11-Pose model weights
            device: Device to run inference on ('cpu', 'cuda', 'mps')
            confidence_threshold: Minimum confidence for detections
        """
        self.model_path = model_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.model = None

        self._load_model()

    def _load_model(self) -> None:
        """Load the YOLOv11-Pose model."""
        if not os.path.exists(self.model_path):
            print(f"Warning: Model weights not found at {self.model_path}")
            print("Using default YOLOv11n-pose model from Ultralytics Hub")
            # Use default model if weights file doesn't exist
            self.model = YOLO("yolov8n-pose.pt")
        else:
            self.model = YOLO(self.model_path)

        # Set device
        self.model.to(self.device)
        print(f"YOLOv11-Pose model loaded on device: {self.device}")

    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Perform pose detection on input image.

        Args:
            image: Input image as numpy array (H, W, C) in BGR format

        Returns:
            List of DetectionResult objects containing pose information
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        # Run inference
        results = self.model(image, conf=self.confidence_threshold, verbose=False)

        detection_results = []

        for i, result in enumerate(results):
            if result.boxes is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confidences = result.boxes.conf.cpu().numpy()

            # Check if pose keypoints are available
            if result.keypoints is None:
                continue

            keypoints = result.keypoints.xy.cpu().numpy()  # [N, 17, 2]
            keypoint_confs = result.keypoints.conf.cpu().numpy()  # [N, 17]

            # Process each detection
            for j in range(len(boxes)):
                bbox = boxes[j]
                confidence = confidences[j]
                kpts = keypoints[j]  # [17, 2]
                kpt_confs = keypoint_confs[j]  # [17]

                # Convert keypoints to Keypoint objects
                keypoint_objects = []
                for k, (kpt, kpt_conf) in enumerate(zip(kpts, kpt_confs)):
                    if k < len(self.KEYPOINT_NAMES):
                        keypoint_obj = Keypoint(
                            x=int(kpt[0]),
                            y=int(kpt[1]),
                            score=float(kpt_conf),
                            label=self.KEYPOINT_NAMES[k],
                        )
                        keypoint_objects.append(keypoint_obj)

                # Create detection result
                detection = DetectionResult(
                    person_id=len(detection_results),  # Simple incrementing ID
                    bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                    keypoints=keypoint_objects,
                    confidence=float(confidence),
                )

                detection_results.append(detection)

        return detection_results

    def get_upper_body_keypoints(self, detection: DetectionResult) -> List[Keypoint]:
        """
        Get upper body keypoints relevant for chest analysis.

        Args:
            detection: Detection result containing keypoints

        Returns:
            List of upper body keypoints
        """
        upper_body_labels = [
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
        ]

        return detection.get_keypoints_by_labels(upper_body_labels)

    def is_valid_detection(
        self, detection: DetectionResult, min_keypoints: int = 5
    ) -> bool:
        """
        Check if a detection has sufficient keypoints for analysis.

        Args:
            detection: Detection result to validate
            min_keypoints: Minimum number of keypoints required

        Returns:
            True if detection is valid for further processing
        """
        valid_keypoints = sum(1 for kp in detection.keypoints if kp.score > 0.3)

        # Check if essential keypoints exist
        essential_labels = ["left_shoulder", "right_shoulder", "nose"]
        essential_count = sum(
            1
            for label in essential_labels
            if detection.get_keypoint_by_label(label) is not None
        )

        return valid_keypoints >= min_keypoints and essential_count >= 2
