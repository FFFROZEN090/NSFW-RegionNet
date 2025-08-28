"""
Core processing pipeline orchestrator for chest exposure analysis.
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import yaml

from .models.yolo_detector import YoloDetector
from .models.sam2_segmenter import SamSegmenter
from .processors.prompt_generator import PromptGenerator
from .processors.chest_analyzer import ChestExposureAnalyzer
from .data_models import DetectionResult
from ..utils.visualization import VisualizationUtils
from ..utils.image_utils import ImageUtils


class ChestExposurePipeline:
    """Main pipeline for processing images through YOLO detection, prompt generation, and SAM2 segmentation."""

    def __init__(self, config_path: str = None):
        """
        Initialize the pipeline with configuration.

        Args:
            config_path: Path to configuration YAML file
        """
        self.config = self._load_config(config_path)

        # Initialize models
        self.yolo_detector = None
        self.sam_segmenter = None
        self.prompt_generator = PromptGenerator()
        self.chest_analyzer = ChestExposureAnalyzer(
            min_intersection_ratio=self.config["exposure_detection"]["min_intersection_ratio"],
            min_intersection_area=self.config["exposure_detection"]["min_intersection_area"],
            morphology_kernel_size=self.config["exposure_detection"]["morphology_kernel_size"],
            opening_iterations=self.config["exposure_detection"]["opening_iterations"],
            closing_iterations=self.config["exposure_detection"]["closing_iterations"],
        )

        # Initialize utilities
        self.visualizer = VisualizationUtils()
        self.image_utils = ImageUtils()

        self._initialize_models()

    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), "../configs/default_config.yaml"
            )

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        return config

    def _initialize_models(self) -> None:
        """Initialize YOLO and SAM2 models."""
        try:
            # Initialize YOLO detector
            yolo_model_path = self.config["models"]["yolo_model_path"]
            confidence_threshold = self.config["detection"]["confidence_threshold"]

            self.yolo_detector = YoloDetector(
                model_path=yolo_model_path,
                device="cpu",  # Can be changed to 'cuda' if available
                confidence_threshold=confidence_threshold,
            )

            # Initialize SAM2 segmenter
            sam2_model_path = self.config["models"]["sam2_model_path"]
            sam2_model_type = self.config["models"].get(
                "sam2_model_type", "hiera_large"
            )

            self.sam_segmenter = SamSegmenter(
                model_path=sam2_model_path,
                model_type=sam2_model_type,
                device="cpu",  # Can be changed to 'cuda' if available
            )

            print("Pipeline models initialized successfully")

        except Exception as e:
            print(f"Error initializing models: {e}")
            print("Models may not be fully functional")

    def process_image(self, image_path: str, output_dir: str = None) -> Dict[str, Any]:
        """
        Process a single image through the complete pipeline.

        Args:
            image_path: Path to input image
            output_dir: Directory to save results (optional)

        Returns:
            Dictionary containing processing results
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        print(f"Processing image: {os.path.basename(image_path)}")
        print(f"Image shape: {image.shape}")

        # Setup output directory
        if output_dir is None:
            output_dir = self.config["paths"]["output_dir"]

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        image_output_dir = os.path.join(output_dir, base_name)
        os.makedirs(image_output_dir, exist_ok=True)

        results = {
            "image_path": image_path,
            "output_dir": image_output_dir,
            "detections": [],
            "segmentation_results": [],
            "exposure_analysis": [],
        }

        # Step 1: YOLO pose detection
        print("Step 1: Running YOLO pose detection...")
        detections = self.yolo_detector.detect(image)
        results["detections"] = detections

        if not detections:
            print("No person detections found")
            return results

        print(f"Found {len(detections)} person detections")

        # Process each detected person
        for i, detection in enumerate(detections):
            person_output_dir = os.path.join(image_output_dir, f"person_{i+1}")
            os.makedirs(person_output_dir, exist_ok=True)

            person_result = self._process_single_detection(
                image, detection, detections, person_output_dir, i + 1
            )

            results["segmentation_results"].append(person_result)

        # Create summary visualization
        if self.config["visualization"]["save_intermediate_steps"]:
            self._create_summary_visualization(image, results, image_output_dir)

        # Step 4: Chest exposure analysis for all persons
        print("Step 4: Analyzing chest exposure...")
        exposure_results = self._analyze_chest_exposure_for_image(image, results)
        results["exposure_analysis"] = exposure_results

        # Step 5: Check if any person shows chest exposure and copy to exposed folder
        if self._should_copy_to_exposed(exposure_results):
            exposed_dir = self.config["paths"]["exposed_dir"]
            self.chest_analyzer.copy_results_to_exposed_folder(
                image_output_dir, exposed_dir, base_name
            )

        return results

    def _process_single_detection(
        self,
        image: np.ndarray,
        detection: DetectionResult,
        all_detections: List[DetectionResult],
        output_dir: str,
        person_id: int,
    ) -> Dict[str, Any]:
        """Process a single person detection through the full pipeline."""

        print(f"Processing Person {person_id}...")

        result = {
            "person_id": person_id,
            "detection": detection,
            "prompts": None,
            "segmentation_mask": None,
            "chest_triangle_mask": None,
            "output_dir": output_dir,
        }

        # Step 2: Generate prompts from keypoints
        print(f"  Step 2: Generating SAM2 prompts...")
        points, labels = self.prompt_generator.generate_prompts(
            detection, (image.shape[0], image.shape[1]), all_detections
        )

        result["prompts"] = {"points": points, "labels": labels}

        if len(points) == 0:
            print(f"  No valid prompts generated for Person {person_id}")
            return result

        print(
            f"  Generated {len(points)} prompt points ({np.sum(labels)} positive, {np.sum(1-labels)} negative)"
        )

        # Step 3: SAM2 segmentation
        print(f"  Step 3: Running SAM2 segmentation...")
        segmentation_mask = self.sam_segmenter.segment(image, points, labels)
        result["segmentation_mask"] = segmentation_mask

        if segmentation_mask is not None:
            mask_area = np.sum(segmentation_mask)
            print(f"  Segmentation completed. Mask area: {mask_area} pixels")
        else:
            print(f"  Segmentation failed for Person {person_id}")

        # Step 3.5: Generate chest triangle mask for analysis
        chest_triangle_mask = self.prompt_generator.generate_chest_triangle_mask(
            detection, (image.shape[0], image.shape[1])
        )
        result["chest_triangle_mask"] = chest_triangle_mask

        # Step 4: Save visualizations
        if self.config["visualization"]["save_intermediate_steps"]:
            self._save_detection_visualizations(image, result, output_dir)

        return result

    def _save_detection_visualizations(
        self, image: np.ndarray, result: Dict[str, Any], output_dir: str
    ) -> None:
        """Save visualization images for a single detection."""

        detection = result["detection"]
        points = result["prompts"]["points"]
        labels = result["prompts"]["labels"]
        mask = result["segmentation_mask"]

        # 1. Keypoints visualization
        keypoints_img = self.visualizer.draw_keypoints(image.copy(), [detection])
        cv2.imwrite(os.path.join(output_dir, "keypoints.png"), keypoints_img)

        # 2. Bounding box visualization
        bbox_img = self.visualizer.draw_bounding_boxes(image.copy(), [detection])
        cv2.imwrite(os.path.join(output_dir, "bounding_box.png"), bbox_img)

        # 3. Prompts visualization
        prompts_img = self.visualizer.draw_prompts(image.copy(), points, labels)
        cv2.imwrite(os.path.join(output_dir, "prompts.png"), prompts_img)

        # 4. Chest triangle (from prompt generator)
        triangle_mask = self.prompt_generator.generate_chest_triangle_mask(
            detection, (image.shape[0], image.shape[1])
        )
        triangle_img = self.visualizer.draw_mask_overlay(
            image.copy(), triangle_mask, color=(0, 255, 255)
        )
        cv2.imwrite(os.path.join(output_dir, "chest_triangle.png"), triangle_img)

        # 5. SAM2 segmentation result
        if mask is not None:
            segmentation_img = self.visualizer.draw_mask_overlay(
                image.copy(), mask, color=(0, 255, 0)
            )
            cv2.imwrite(
                os.path.join(output_dir, "sam2_segmentation.png"), segmentation_img
            )

            # Combined visualization
            combined_img = self.visualizer.create_combined_visualization(
                image, detection, points, labels, mask
            )
            cv2.imwrite(os.path.join(output_dir, "combined.png"), combined_img)

        print(f"  Visualizations saved to: {output_dir}")

    def _create_summary_visualization(
        self, image: np.ndarray, results: Dict[str, Any], output_dir: str
    ) -> None:
        """Create a summary visualization showing all detections and segmentations."""

        summary_img = image.copy()

        # Draw all detections
        detections = results["detections"]
        summary_img = self.visualizer.draw_keypoints(summary_img, detections)
        summary_img = self.visualizer.draw_bounding_boxes(summary_img, detections)

        # Draw all segmentation masks
        for i, seg_result in enumerate(results["segmentation_results"]):
            if seg_result["segmentation_mask"] is not None:
                # Use different colors for different people
                colors = [
                    (0, 255, 0),
                    (255, 0, 0),
                    (0, 0, 255),
                    (255, 255, 0),
                    (255, 0, 255),
                ]
                color = colors[i % len(colors)]

                summary_img = self.visualizer.draw_mask_overlay(
                    summary_img, seg_result["segmentation_mask"], color=color, alpha=0.3
                )

        # Save summary
        cv2.imwrite(os.path.join(output_dir, "summary.png"), summary_img)
        print(f"Summary visualization saved: {os.path.join(output_dir, 'summary.png')}")

    def process_batch(
        self, input_dir: str, output_dir: str = None
    ) -> List[Dict[str, Any]]:
        """
        Process all images in a directory.

        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save results (optional)

        Returns:
            List of processing results for each image
        """
        if output_dir is None:
            output_dir = self.config["paths"]["output_dir"]

        os.makedirs(output_dir, exist_ok=True)

        # Get all image files
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        image_files = []

        for file_name in os.listdir(input_dir):
            if any(file_name.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(input_dir, file_name))

        if not image_files:
            print(f"No image files found in {input_dir}")
            return []

        print(f"Processing {len(image_files)} images from {input_dir}")

        all_results = []
        for image_path in image_files:
            try:
                result = self.process_image(image_path, output_dir)
                all_results.append(result)
                print(f"Successfully processed: {os.path.basename(image_path)}")
            except Exception as e:
                print(f"Error processing {os.path.basename(image_path)}: {e}")
                continue

        print(
            f"Batch processing complete. Processed {len(all_results)}/{len(image_files)} images"
        )
        return all_results

    def _analyze_chest_exposure_for_image(
        self, image: np.ndarray, results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Analyze chest exposure for all persons in the image.

        Args:
            image: Original input image
            results: Processing results containing segmentation data

        Returns:
            List of exposure analysis results for each person
        """
        exposure_results = []

        for seg_result in results["segmentation_results"]:
            # Get masks
            sam2_mask = seg_result.get("segmentation_mask")
            chest_mask = seg_result.get("chest_triangle_mask")
            detection = seg_result.get("detection")

            # Skip if essential components are missing
            if sam2_mask is None or chest_mask is None or detection is None:
                continue

            # Perform exposure analysis
            analysis_result = self.chest_analyzer.analyze_chest_exposure(
                sam2_mask, chest_mask, detection
            )

            # Save exposure visualization if requested
            if self.config["exposure_detection"]["save_exposure_analysis"]:
                exposure_vis = self.chest_analyzer.create_exposure_visualization(
                    image, analysis_result, sam2_mask, chest_mask
                )

                # Save exposure analysis image
                output_dir = seg_result["output_dir"]
                exposure_path = os.path.join(output_dir, "exposure_analysis.png")
                cv2.imwrite(exposure_path, exposure_vis)

                # Save morphological processing comparison if mask3 data available
                if "mask3_raw" in analysis_result and "mask3_refined" in analysis_result:
                    morphology_comparison = self.chest_analyzer.create_morphology_comparison_visualization(
                        image, 
                        analysis_result["mask3_raw"], 
                        analysis_result["mask3_refined"]
                    )
                    morphology_path = os.path.join(output_dir, "morphology_comparison.png")
                    cv2.imwrite(morphology_path, morphology_comparison)

            # Add to results
            exposure_results.append(analysis_result)

            # Print analysis summary
            person_id = detection.person_id
            is_exposed = analysis_result["is_exposed"]
            confidence = analysis_result["analysis_confidence"]
            intersection_area = analysis_result["intersection_area"]

            status = "EXPOSED" if is_exposed else "NOT EXPOSED"
            print(f"  Person {person_id}: {status} (confidence: {confidence:.2f}, intersection: {intersection_area}px)")

        return exposure_results

    def _should_copy_to_exposed(self, exposure_results: List[Dict[str, Any]]) -> bool:
        """
        Determine if image should be copied to exposed folder.

        Args:
            exposure_results: List of exposure analysis results

        Returns:
            True if any person shows confident chest exposure
        """
        min_confidence = self.config["exposure_detection"]["min_confidence_threshold"]
        should_copy = self.chest_analyzer.should_copy_to_exposed_folder(
            exposure_results, min_confidence
        )

        if should_copy:
            print("  → Image contains chest exposure, copying to exposed folder")

        return should_copy

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline configuration and models."""
        return {
            "config": self.config,
            "yolo_detector": self.yolo_detector is not None,
            "sam_segmenter": (
                self.sam_segmenter.get_model_info() if self.sam_segmenter else None
            ),
            "models_initialized": self.yolo_detector is not None
            and self.sam_segmenter is not None,
        }
