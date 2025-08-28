#!/usr/bin/env python3
"""
Test script for pose detection and prompt generation pipeline.

This script tests the YoloDetector and PromptGenerator modules with images
from data/input directory and saves visualization results to data/output.
"""

import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from chest_exposure_analyzer.core.models.yolo_detector import YoloDetector
from chest_exposure_analyzer.core.processors.prompt_generator import PromptGenerator
from chest_exposure_analyzer.utils.visualization import VisualizationUtils


def load_test_images(input_dir: str) -> list:
    """
    Load test images from input directory.
    
    Args:
        input_dir: Directory containing test images
        
    Returns:
        List of (image_path, image) tuples
    """
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        return []
    
    for file_path in Path(input_dir).iterdir():
        if file_path.suffix.lower() in supported_formats:
            image = cv2.imread(str(file_path))
            if image is not None:
                image_files.append((str(file_path), image))
                print(f"Loaded: {file_path.name} - {image.shape}")
            else:
                print(f"Failed to load: {file_path.name}")
    
    return image_files


def test_pose_detection_pipeline(test_images: list, output_dir: str) -> None:
    """
    Test the pose detection pipeline with loaded images.
    
    Args:
        test_images: List of (image_path, image) tuples
        output_dir: Directory to save results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize detector (will use default model if weights don't exist)
    print("\\nInitializing YOLOv11-Pose detector...")
    detector = YoloDetector(
        model_path="weights/yolov11l-pose.pt",  # Will fallback to default if not found
        device='cpu',  # Use CPU for compatibility
        confidence_threshold=0.3
    )
    
    print(f"\\nProcessing {len(test_images)} test images...")
    
    for i, (image_path, image) in enumerate(test_images):
        print(f"\\n--- Processing Image {i+1}/{len(test_images)}: {Path(image_path).name} ---")
        
        # Start timing
        start_time = time.time()
        
        # Run pose detection
        detections = detector.detect(image)
        detection_time = time.time() - start_time
        
        print(f"Detection completed in {detection_time:.3f}s")
        print(f"Found {len(detections)} person(s)")
        
        # Create base filename for outputs
        base_name = Path(image_path).stem
        
        if not detections:
            print("No persons detected, skipping further processing")
            # Save original image to output
            output_path = os.path.join(output_dir, f"{base_name}_no_detection.png")
            cv2.imwrite(output_path, image)
            continue
        
        # Process each detection
        for j, detection in enumerate(detections):
            print(f"\\n  Person {j+1} (ID: {detection.person_id}):")
            print(f"    Confidence: {detection.confidence:.3f}")
            print(f"    Bbox: {detection.bbox}")
            print(f"    Keypoints: {len(detection.keypoints)}")
            
            # Validate detection
            if not detector.is_valid_detection(detection):
                print("    Warning: Detection has insufficient keypoints, skipping")
                continue
            
            # Print keypoint details
            valid_keypoints = [kp for kp in detection.keypoints if kp.score > 0.3]
            print(f"    Valid keypoints ({len(valid_keypoints)}):")
            for kp in valid_keypoints:
                print(f"      {kp.label}: ({kp.x}, {kp.y}) conf={kp.score:.3f}")
            
            # Generate SAM2 prompts (pass all detections for safe negative point placement)
            print("    Generating SAM2 prompts...")
            prompts_points, prompts_labels = PromptGenerator.generate_prompts(
                detection, image.shape[:2], all_detections=detections
            )
            
            positive_count = np.sum(prompts_labels == 1)
            negative_count = np.sum(prompts_labels == 0)
            print(f"    Generated {len(prompts_points)} prompts: {positive_count} positive, {negative_count} negative")
            
            # Generate chest triangle mask
            print("    Generating chest triangle mask...")
            chest_mask = PromptGenerator.generate_chest_triangle_mask(
                detection, image.shape[:2]
            )
            
            chest_area = np.sum(chest_mask > 0)
            print(f"    Chest triangle area: {chest_area} pixels")
            
            # Create visualizations
            print("    Creating visualizations...")
            person_output_dir = os.path.join(output_dir, f"{base_name}_person_{j+1}")
            
            visualizations = VisualizationUtils.create_pipeline_visualization(
                original_image=image,
                detection=detection,
                prompts_points=prompts_points,
                prompts_labels=prompts_labels,
                chest_mask=chest_mask,
                save_dir=person_output_dir
            )
            
            print(f"    Saved {len(visualizations)} visualizations to: {person_output_dir}")
        
        # Create summary visualization with all detections
        print("\\n  Creating summary visualization...")
        summary_img = VisualizationUtils.create_detection_summary(
            image, detections, 
            save_path=os.path.join(output_dir, f"{base_name}_summary.png")
        )
        
        total_time = time.time() - start_time
        print(f"\\n  Total processing time: {total_time:.3f}s")


def create_sample_test_data(input_dir: str) -> None:
    """
    Create some sample test data if input directory is empty.
    
    Args:
        input_dir: Directory to create sample data in
    """
    os.makedirs(input_dir, exist_ok=True)
    
    # Check if directory is empty
    existing_files = list(Path(input_dir).glob("*"))
    if existing_files:
        return  # Directory has files, don't create samples
    
    print(f"Creating sample test image in {input_dir}...")
    
    # Create a simple test image with a basic human-like figure
    img = np.ones((600, 400, 3), dtype=np.uint8) * 240  # Light gray background
    
    # Draw a simple stick figure for testing
    # Head
    cv2.circle(img, (200, 100), 30, (100, 150, 200), -1)
    
    # Body
    cv2.line(img, (200, 130), (200, 350), (100, 150, 200), 8)
    
    # Arms
    cv2.line(img, (200, 180), (150, 250), (100, 150, 200), 6)
    cv2.line(img, (200, 180), (250, 250), (100, 150, 200), 6)
    
    # Legs
    cv2.line(img, (200, 350), (170, 450), (100, 150, 200), 6)
    cv2.line(img, (200, 350), (230, 450), (100, 150, 200), 6)
    
    # Add some text
    cv2.putText(img, "Sample Test Image", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2)
    cv2.putText(img, "Replace with real images", (50, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
    
    # Save sample image
    sample_path = os.path.join(input_dir, "sample_test.png")
    cv2.imwrite(sample_path, img)
    print(f"Created sample test image: {sample_path}")
    print("Please add real test images to the input directory for better results!")


def main():
    """Main function to run pose detection tests."""
    print("=== NSFW-RegionNet Pose Detection Test ===")
    print("Testing YOLOv11-Pose detection and prompt generation pipeline\\n")
    
    # Define paths
    input_dir = "chest_exposure_analyzer/data/input"
    output_dir = "chest_exposure_analyzer/data/output"
    
    # Create sample data if needed
    create_sample_test_data(input_dir)
    
    # Load test images
    print("Loading test images...")
    test_images = load_test_images(input_dir)
    
    if not test_images:
        print(f"No valid images found in {input_dir}")
        print("Please add test images (.jpg, .png, etc.) to the input directory")
        return
    
    # Run tests
    try:
        test_pose_detection_pipeline(test_images, output_dir)
        print(f"\\n=== Test completed successfully! ===")
        print(f"Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"\\nError during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()