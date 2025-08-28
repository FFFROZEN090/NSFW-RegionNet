"""
Chest region analysis and mask processing.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import cv2
import os
import shutil

from ..data_models import DetectionResult


class ChestExposureAnalyzer:
    """Analyzes chest exposure by detecting intersection between skin and chest regions."""

    def __init__(
        self,
        min_intersection_ratio: float = 0.01,
        min_intersection_area: int = 100,
        morphology_kernel_size: int = 5,
        opening_iterations: int = 1,
        closing_iterations: int = 2,
    ):
        """
        Initialize chest exposure analyzer.

        Args:
            min_intersection_ratio: Minimum ratio of intersection to chest area to consider exposed
            min_intersection_area: Minimum intersection area in pixels to consider exposed
            morphology_kernel_size: Size of morphological operation kernel
            opening_iterations: Number of opening iterations to remove noise
            closing_iterations: Number of closing iterations to fill gaps
        """
        self.min_intersection_ratio = min_intersection_ratio
        self.min_intersection_area = min_intersection_area
        self.morphology_kernel_size = morphology_kernel_size
        self.opening_iterations = opening_iterations
        self.closing_iterations = closing_iterations

    def analyze_chest_exposure(
        self,
        sam2_skin_mask: np.ndarray,
        chest_triangle_mask: np.ndarray,
        detection: DetectionResult,
    ) -> Dict[str, Any]:
        """
        Analyze chest exposure by detecting intersection between skin and chest triangle masks.

        Args:
            sam2_skin_mask: SAM2 segmented skin region mask (boolean array)
            chest_triangle_mask: Chest triangle geometry mask (boolean array)
            detection: Detection result containing keypoints

        Returns:
            Dictionary containing exposure analysis results
        """
        # Ensure masks are boolean
        if sam2_skin_mask.dtype != bool:
            skin_mask = sam2_skin_mask > 0
        else:
            skin_mask = sam2_skin_mask

        if chest_triangle_mask.dtype != bool:
            chest_mask = chest_triangle_mask > 0
        else:
            chest_mask = chest_triangle_mask

        # Step 1: Calculate raw intersection (mask3)
        raw_intersection_mask = skin_mask & chest_mask
        
        # Step 2: Create mask3 - extract skin regions that intersect with chest area
        mask3 = skin_mask.copy()
        mask3 = mask3 & chest_mask  # Only keep skin pixels that are within chest triangle
        
        # Step 3: Apply morphological operations to mask3 for cleaner geometry
        refined_exposure_mask = self._apply_morphological_operations(mask3)
        
        # Use refined mask for final calculations
        intersection_mask = refined_exposure_mask
        intersection_area = np.sum(intersection_mask)

        # Calculate areas
        skin_area = np.sum(skin_mask)
        chest_area = np.sum(chest_mask)

        # Calculate ratios
        intersection_to_chest_ratio = (
            intersection_area / chest_area if chest_area > 0 else 0.0
        )
        intersection_to_skin_ratio = (
            intersection_area / skin_area if skin_area > 0 else 0.0
        )

        # Determine if chest is exposed
        is_exposed = (
            intersection_area >= self.min_intersection_area
            and intersection_to_chest_ratio >= self.min_intersection_ratio
        )

        # Create analysis result
        analysis_result = {
            "is_exposed": is_exposed,
            "intersection_area": int(intersection_area),
            "skin_area": int(skin_area),
            "chest_area": int(chest_area),
            "intersection_to_chest_ratio": float(intersection_to_chest_ratio),
            "intersection_to_skin_ratio": float(intersection_to_skin_ratio),
            "intersection_mask": intersection_mask,
            "mask3_raw": mask3,
            "mask3_refined": refined_exposure_mask,
            "person_id": detection.person_id,
            "confidence": detection.confidence,
            "analysis_confidence": self._calculate_analysis_confidence(
                detection, intersection_to_chest_ratio, skin_area, chest_area
            ),
        }

        return analysis_result

    def _apply_morphological_operations(self, mask3: np.ndarray) -> np.ndarray:
        """
        Apply morphological opening and closing operations to mask3 for cleaner geometry.
        
        Args:
            mask3: Binary mask of chest exposure region (intersection of skin and chest triangle)
            
        Returns:
            Refined mask with cleaner geometric regions after morphological processing
        """
        if mask3.dtype != np.uint8:
            # Convert boolean mask to uint8 for OpenCV operations
            mask3_uint8 = (mask3 * 255).astype(np.uint8)
        else:
            mask3_uint8 = mask3.copy()
            
        # Create morphological kernel
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morphology_kernel_size, self.morphology_kernel_size)
        )
        
        # Step 1: Opening operation - removes noise and small objects
        # Opening = Erosion followed by Dilation
        opened_mask = cv2.morphologyEx(
            mask3_uint8,
            cv2.MORPH_OPEN,
            kernel,
            iterations=self.opening_iterations
        )
        
        # Step 2: Closing operation - fills gaps and connects nearby regions
        # Closing = Dilation followed by Erosion
        refined_mask = cv2.morphologyEx(
            opened_mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=self.closing_iterations
        )
        
        # Convert back to boolean for consistency
        return refined_mask > 0

    def _calculate_analysis_confidence(
        self,
        detection: DetectionResult,
        intersection_ratio: float,
        skin_area: int,
        chest_area: int,
    ) -> float:
        """
        Calculate confidence score for the exposure analysis.

        Args:
            detection: Detection result
            intersection_ratio: Ratio of intersection to chest area
            skin_area: Area of detected skin
            chest_area: Area of chest triangle

        Returns:
            Confidence score between 0 and 1
        """
        # Base confidence from person detection
        base_confidence = detection.confidence

        # Key keypoints confidence (shoulders, chest area related)
        key_keypoints = ["left_shoulder", "right_shoulder", "nose"]
        keypoint_confidence = 0.0
        valid_keypoints = 0

        for kp_label in key_keypoints:
            kp = detection.get_keypoint_by_label(kp_label)
            if kp and kp.score > 0.3:
                keypoint_confidence += kp.score
                valid_keypoints += 1

        keypoint_confidence = (
            keypoint_confidence / len(key_keypoints) if valid_keypoints > 0 else 0.0
        )

        # Area-based confidence (larger areas are more reliable)
        area_confidence = min(1.0, (skin_area + chest_area) / 10000)  # Normalize to 10k pixels

        # Intersection confidence (clearer intersections are more reliable)
        intersection_confidence = min(1.0, intersection_ratio * 10)  # Scale up ratio

        # Weighted average
        weights = [0.3, 0.3, 0.2, 0.2]  # base, keypoints, area, intersection
        confidences = [
            base_confidence,
            keypoint_confidence,
            area_confidence,
            intersection_confidence,
        ]

        final_confidence = sum(w * c for w, c in zip(weights, confidences))
        return max(0.0, min(1.0, final_confidence))

    def create_exposure_visualization(
        self,
        original_image: np.ndarray,
        analysis_result: Dict[str, Any],
        sam2_mask: np.ndarray,
        chest_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Create visualization showing chest exposure analysis.

        Args:
            original_image: Original input image
            analysis_result: Results from analyze_chest_exposure
            sam2_mask: SAM2 skin segmentation mask
            chest_mask: Chest triangle mask

        Returns:
            Visualization image showing exposure analysis
        """
        # Create base image
        vis_image = original_image.copy()

        # Draw chest triangle (blue, semi-transparent)
        chest_overlay = np.zeros_like(original_image)
        chest_overlay[chest_mask] = [255, 165, 0]  # Orange for chest area
        vis_image = cv2.addWeighted(vis_image, 0.8, chest_overlay, 0.2, 0)

        # Draw SAM2 skin mask (green, semi-transparent)
        skin_overlay = np.zeros_like(original_image)
        skin_overlay[sam2_mask > 0] = [0, 255, 0]  # Green for skin
        vis_image = cv2.addWeighted(vis_image, 0.8, skin_overlay, 0.2, 0)

        # Draw refined intersection (red, more opaque for emphasis)
        intersection_mask = analysis_result["intersection_mask"]
        intersection_overlay = np.zeros_like(original_image)
        intersection_overlay[intersection_mask] = [0, 0, 255]  # Red for refined intersection
        vis_image = cv2.addWeighted(vis_image, 0.7, intersection_overlay, 0.3, 0)

        # Add text information
        is_exposed = analysis_result["is_exposed"]
        confidence = analysis_result["analysis_confidence"]
        intersection_area = analysis_result["intersection_area"]
        ratio = analysis_result["intersection_to_chest_ratio"]

        # Status text
        status_text = "EXPOSED" if is_exposed else "NOT EXPOSED"
        status_color = (0, 0, 255) if is_exposed else (0, 255, 0)

        cv2.putText(
            vis_image,
            f"Status: {status_text}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2,
        )

        # Analysis details
        details = [
            f"Confidence: {confidence:.2f}",
            f"Intersection: {intersection_area}px",
            f"Ratio: {ratio:.3f}",
            f"Person ID: {analysis_result['person_id']}",
        ]

        for i, detail in enumerate(details):
            cv2.putText(
                vis_image,
                detail,
                (20, 80 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

        return vis_image

    def create_morphology_comparison_visualization(
        self,
        original_image: np.ndarray,
        mask3_raw: np.ndarray,
        mask3_refined: np.ndarray,
    ) -> np.ndarray:
        """
        Create comparison visualization showing before/after morphological operations.

        Args:
            original_image: Original input image
            mask3_raw: Raw intersection mask before morphological operations
            mask3_refined: Refined mask after opening and closing operations

        Returns:
            Side-by-side comparison visualization
        """
        h, w = original_image.shape[:2]
        
        # Create side-by-side comparison
        comparison_img = np.zeros((h, w * 2, 3), dtype=np.uint8)
        
        # Left side: Original image with raw mask3
        left_img = original_image.copy()
        raw_overlay = np.zeros_like(original_image)
        raw_overlay[mask3_raw] = [0, 255, 255]  # Cyan for raw intersection
        left_img = cv2.addWeighted(left_img, 0.7, raw_overlay, 0.3, 0)
        comparison_img[:, :w] = left_img
        
        # Right side: Original image with refined mask3
        right_img = original_image.copy()
        refined_overlay = np.zeros_like(original_image)
        refined_overlay[mask3_refined] = [255, 0, 255]  # Magenta for refined intersection
        right_img = cv2.addWeighted(right_img, 0.7, refined_overlay, 0.3, 0)
        comparison_img[:, w:] = right_img
        
        # Add labels
        cv2.putText(
            comparison_img,
            "Raw Intersection (Mask3)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        
        cv2.putText(
            comparison_img,
            "Refined Intersection (Open+Close)",
            (w + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 255),
            2,
        )
        
        # Add area statistics
        raw_area = np.sum(mask3_raw)
        refined_area = np.sum(mask3_refined)
        
        cv2.putText(
            comparison_img,
            f"Area: {raw_area}px",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        
        cv2.putText(
            comparison_img,
            f"Area: {refined_area}px",
            (w + 10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        
        return comparison_img

    def should_copy_to_exposed_folder(
        self, analysis_results: list, min_confidence: float = 0.5
    ) -> bool:
        """
        Determine if image should be copied to exposed folder based on analysis results.

        Args:
            analysis_results: List of analysis results for all persons in image
            min_confidence: Minimum confidence threshold for exposure

        Returns:
            True if any person shows confident chest exposure
        """
        for result in analysis_results:
            if (
                result["is_exposed"]
                and result["analysis_confidence"] >= min_confidence
            ):
                return True
        return False

    def copy_results_to_exposed_folder(
        self, source_dir: str, exposed_dir: str, image_name: str
    ) -> None:
        """
        Copy all processing results to exposed folder.

        Args:
            source_dir: Source directory containing processing results
            exposed_dir: Target exposed results directory  
            image_name: Base name of the image
        """
        # Create exposed directory if it doesn't exist
        os.makedirs(exposed_dir, exist_ok=True)

        # Create subdirectory for this image
        image_exposed_dir = os.path.join(exposed_dir, image_name)
        os.makedirs(image_exposed_dir, exist_ok=True)

        # Copy all contents from source directory
        if os.path.exists(source_dir):
            try:
                # Copy the entire directory tree
                for item in os.listdir(source_dir):
                    source_item = os.path.join(source_dir, item)
                    target_item = os.path.join(image_exposed_dir, item)

                    if os.path.isdir(source_item):
                        shutil.copytree(source_item, target_item, dirs_exist_ok=True)
                    else:
                        shutil.copy2(source_item, target_item)

                print(f"  Copied exposure results to: {image_exposed_dir}")

            except Exception as e:
                print(f"  Error copying exposure results: {e}")
        else:
            print(f"  Warning: Source directory not found: {source_dir}")
