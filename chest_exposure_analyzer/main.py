"""
Main entry point for the chest exposure analyzer.
"""

import argparse
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from chest_exposure_analyzer.core.pipeline import ChestExposurePipeline


def run_demo():
    """Run demo mode with sample images."""
    print("Running NSFW-RegionNet Demo")
    print("=" * 50)

    # Initialize pipeline
    try:
        pipeline = ChestExposurePipeline()
        print("Pipeline initialized successfully")
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        return 1

    # Look for demo images
    input_dir = "chest_exposure_analyzer/data/input"
    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        print("Please add demo images to the input directory")
        return 1

    # Find available images
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    demo_images = []

    for ext in image_extensions:
        demo_images.extend(Path(input_dir).glob(f"*{ext}"))
        demo_images.extend(Path(input_dir).glob(f"*{ext.upper()}"))

    if not demo_images:
        print(f"No demo images found in {input_dir}")
        print("Supported formats: jpg, jpeg, png, bmp, tiff")
        return 1

    print(f"Found {len(demo_images)} demo image(s)")

    # Process each demo image
    for i, image_path in enumerate(demo_images[:7], 1):  # Limit to 3 images for demo
        print(f"\nProcessing image {i}: {image_path.name}")
        try:
            results = pipeline.process_image(str(image_path))
            print(f"Results saved to: {results['output_dir']}")

            # Show summary
            num_detections = len(results.get("detections", []))
            print(f"Found {num_detections} person(s)")

        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")
            continue

    print(f"\nDemo completed! Check the output directory for results.")
    return 0


def run_single_image(image_path: str, output_dir: str = None):
    """Process a single image."""
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return 1

    print(f"Processing: {image_path}")

    try:
        pipeline = ChestExposurePipeline()
        results = pipeline.process_image(image_path, output_dir)

        print(f"Processing completed")
        print(f"Found {len(results.get('detections', []))} person(s)")
        print(f"Results saved to: {results['output_dir']}")

        return 0

    except Exception as e:
        print(f"Error processing image: {e}")
        return 1


def main():
    """Main function to run the chest exposure analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="NSFW-RegionNet: Chest Exposure Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python chest_exposure_analyzer/main.py --demo
  python chest_exposure_analyzer/main.py --image path/to/image.jpg
  python chest_exposure_analyzer/main.py --image path/to/image.jpg --output custom_output/
        """,
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo mode with sample images from input directory",
    )

    parser.add_argument("--image", "-i", type=str, help="Path to input image file")

    parser.add_argument(
        "--output", "-o", type=str, help="Custom output directory (optional)"
    )

    parser.add_argument(
        "--version", "-v", action="version", version="NSFW-RegionNet v1.0.0"
    )

    args = parser.parse_args()

    # Show help if no arguments provided
    if not any([args.demo, args.image]):
        parser.print_help()
        return 0

    # Run demo mode
    if args.demo:
        return run_demo()

    # Process single image
    if args.image:
        return run_single_image(args.image, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
