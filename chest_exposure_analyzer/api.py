"""
Convenient API module for chest exposure analysis.
Simple interface for processing single images.
"""

from typing import Dict, Any
from .core.pipeline import ChestExposurePipeline


def process_image_for_content_filtering(
    image_path: str, output_dir: str = None
) -> Dict[str, Any]:
    """
    Process a single image for content filtering with automatic mosaic application.
    
    This is the main API function for content filtering applications.
    
    Args:
        image_path: Path to input image file
        output_dir: Output directory (optional, defaults to same dir as input)
        
    Returns:
        Dictionary containing:
        - needs_mosaic: bool - Whether image contains chest exposure
        - original_path: str - Path to original image 
        - output_path: str - Path to processed image
        - status: str - "ORIGINAL" or "MOSAICKED"
        - person_count: int - Number of persons detected
        - frontal_person_count: int - Number of frontal persons analyzed
        
    Example:
        >>> from chest_exposure_analyzer.api import process_image_for_content_filtering
        >>> result = process_image_for_content_filtering("/path/to/image.jpg")
        >>> print(f"Needs filtering: {result['needs_mosaic']}")
        >>> print(f"Output saved to: {result['output_path']}")
    """
    # Initialize pipeline with default configuration
    pipeline = ChestExposurePipeline()
    
    # Process the image
    return pipeline.process_single_image_file(image_path, output_dir)


def create_content_filter_pipeline() -> ChestExposurePipeline:
    """
    Create and return a configured pipeline instance for batch processing.
    
    Returns:
        ChestExposurePipeline instance ready for processing
        
    Example:
        >>> from chest_exposure_analyzer.api import create_content_filter_pipeline
        >>> pipeline = create_content_filter_pipeline()
        >>> result = pipeline.process_single_image_file("/path/to/image.jpg")
    """
    return ChestExposurePipeline()


# Convenience aliases for common use cases
analyze_image = process_image_for_content_filtering
filter_image = process_image_for_content_filtering