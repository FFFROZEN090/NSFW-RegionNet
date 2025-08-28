"""
Data models and contracts for the chest exposure analyzer.
"""

from typing import List, Tuple, Optional
import numpy as np
from pydantic import BaseModel, Field


class Keypoint(BaseModel):
    """Represents a detected keypoint with position, confidence score, and label."""
    x: int = Field(..., description="X coordinate of the keypoint")
    y: int = Field(..., description="Y coordinate of the keypoint")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score of the keypoint")
    label: str = Field(..., description="Label of the keypoint (e.g., 'nose', 'left_shoulder')")

    class Config:
        """Pydantic model configuration."""
        json_encoders = {
            np.ndarray: lambda v: v.tolist()
        }


class DetectionResult(BaseModel):
    """Represents the result of person detection with bounding box and keypoints."""
    person_id: int = Field(..., description="Unique identifier for the detected person")
    bbox: Tuple[int, int, int, int] = Field(..., description="Bounding box coordinates (x1, y1, x2, y2)")
    keypoints: List[Keypoint] = Field(default_factory=list, description="List of detected keypoints")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall detection confidence")

    def get_keypoint_by_label(self, label: str) -> Optional[Keypoint]:
        """Get a keypoint by its label."""
        for keypoint in self.keypoints:
            if keypoint.label == label:
                return keypoint
        return None

    def get_keypoints_by_labels(self, labels: List[str]) -> List[Keypoint]:
        """Get multiple keypoints by their labels."""
        return [kp for kp in self.keypoints if kp.label in labels]

    class Config:
        """Pydantic model configuration."""
        json_encoders = {
            np.ndarray: lambda v: v.tolist()
        }


class PipelineResult(BaseModel):
    """Represents the complete result of the chest exposure analysis pipeline."""
    original_image: Optional[np.ndarray] = Field(default=None, description="Original input image")
    image_shape: Tuple[int, int, int] = Field(..., description="Shape of the input image (H, W, C)")
    detection_results: List[DetectionResult] = Field(default_factory=list, description="Person detection results")
    
    # Mask results
    skin_mask: Optional[np.ndarray] = Field(default=None, description="Skin segmentation mask from SAM2")
    chest_area_mask: Optional[np.ndarray] = Field(default=None, description="Chest area mask from geometry")
    final_exposed_mask: Optional[np.ndarray] = Field(default=None, description="Final exposed area mask")
    
    # Analysis results
    is_exposed: bool = Field(default=False, description="Whether exposed chest area is detected")
    exposed_area: float = Field(default=0.0, ge=0.0, description="Area of exposed region in pixels")
    exposure_ratio: float = Field(default=0.0, ge=0.0, le=1.0, description="Ratio of exposed area to total chest area")
    
    # Processing metadata
    processing_time: float = Field(default=0.0, description="Total processing time in seconds")
    model_versions: dict = Field(default_factory=dict, description="Versions of models used")

    class Config:
        """Pydantic model configuration."""
        arbitrary_types_allowed = True
        json_encoders = {
            np.ndarray: lambda v: v.tolist() if v is not None else None
        }

    def get_detection_by_id(self, person_id: int) -> Optional[DetectionResult]:
        """Get detection result by person ID."""
        for detection in self.detection_results:
            if detection.person_id == person_id:
                return detection
        return None

    def has_valid_detections(self) -> bool:
        """Check if there are valid detection results."""
        return len(self.detection_results) > 0

    def get_summary_stats(self) -> dict:
        """Get summary statistics of the analysis."""
        return {
            "num_persons_detected": len(self.detection_results),
            "is_exposed": self.is_exposed,
            "exposed_area_pixels": self.exposed_area,
            "exposure_ratio": self.exposure_ratio,
            "processing_time_ms": self.processing_time * 1000,
            "image_dimensions": self.image_shape[:2]  # H, W
        }


class AnalysisConfig(BaseModel):
    """Configuration for chest exposure analysis parameters."""
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    max_detections: int = Field(default=10, ge=1)
    
    # Chest analysis parameters
    chest_triangle_expansion_factor: float = Field(default=1.2, gt=0.0)
    min_connected_component_area: int = Field(default=100, ge=0)
    exposure_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    
    # SAM2 parameters
    points_per_side: int = Field(default=32, ge=1)
    pred_iou_thresh: float = Field(default=0.88, ge=0.0, le=1.0)
    stability_score_thresh: float = Field(default=0.95, ge=0.0, le=1.0)
    multimask_output: bool = Field(default=True)

    class Config:
        """Pydantic model configuration."""
        json_encoders = {
            np.ndarray: lambda v: v.tolist()
        }