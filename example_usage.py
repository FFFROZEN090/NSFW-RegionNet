#!/usr/bin/env python3
"""
Example usage of the chest exposure analyzer API.
Demonstrates how to use the simple API for content filtering.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chest_exposure_analyzer.api import process_image_for_content_filtering


def main():
    """Example usage of the API."""
    
    # Example 1: Process a single image with default output location
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
        print("Processing image with content filter...")
        print("=" * 50)
        
        try:
            result = process_image_for_content_filtering(image_path)
            
            print(f"Analysis Results:")
            print(f"   Input: {result['original_path']}")
            print(f"   Output: {result['output_path']}")
            print(f"   Status: {result['status']}")
            print(f"   Needs Mosaic: {'Yes' if result['needs_mosaic'] else 'No'}")
            print(f"   Persons Detected: {result['person_count']}")
            print(f"   Frontal Persons Analyzed: {result['frontal_person_count']}")
            
            if result['needs_mosaic']:
                print("Image contains chest exposure - mosaic applied")
            else:
                print("Image is clean - original image copied")
                
        except Exception as e:
            print(f"Error processing image: {e}")
            
    else:
        print("Usage Examples:")
        print(f"   {sys.argv[0]} /path/to/image.jpg")
        print()
        print("Programmatic Usage:")
        print("""
from chest_exposure_analyzer.api import process_image_for_content_filtering

# Process single image
result = process_image_for_content_filtering("/path/to/image.jpg")

if result['needs_mosaic']:
    print(f"Filtered image saved to: {result['output_path']}")
else:
    print("Clean image - no filtering needed")
""")

        # Example with demo images if they exist
        demo_dir = "chest_exposure_analyzer/data/input"
        if os.path.exists(demo_dir):
            print(f"Try with demo images:")
            for filename in os.listdir(demo_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    print(f"   python {sys.argv[0]} {demo_dir}/{filename}")
                    break


if __name__ == "__main__":
    main()