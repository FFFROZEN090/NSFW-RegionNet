#!/usr/bin/env python3
"""
Setup script for NSFW-RegionNet deployment.
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(command, description=""):
    """Run a shell command and handle errors."""
    print(f"📦 {description}...")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print("Error:", e.stderr)
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major != 3 or version.minor < 8:
        print(f"❌ Python 3.8+ required, but found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def create_virtual_environment():
    """Create virtual environment if it doesn't exist."""
    venv_path = Path("NSFW-RegionNet")
    
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    return run_command(
        "python -m venv NSFW-RegionNet", 
        "Creating virtual environment"
    )


def install_dependencies():
    """Install all required dependencies."""
    commands = [
        ("source NSFW-RegionNet/bin/activate && pip install --upgrade pip", "Upgrading pip"),
        ("source NSFW-RegionNet/bin/activate && pip install -r requirements.txt", "Installing dependencies from requirements.txt"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    
    return True


def download_model_weights():
    """Download model weights if they don't exist."""
    weights_dir = Path("chest_exposure_analyzer/weights")
    sam2_weights = weights_dir / "sam2_hiera_large.pt"
    
    # Create weights directory
    weights_dir.mkdir(exist_ok=True)
    
    # Download SAM2 weights if not present
    if not sam2_weights.exists():
        download_command = (
            f"curl -L https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt "
            f"-o {sam2_weights}"
        )
        if not run_command(download_command, "Downloading SAM2 model weights"):
            return False
    else:
        print("✅ SAM2 model weights already exist")
    
    return True


def test_installation():
    """Test if the installation works correctly."""
    test_commands = [
        (
            "source NSFW-RegionNet/bin/activate && python -c 'import torch; print(f\"PyTorch: {torch.__version__}\")'",
            "Testing PyTorch installation"
        ),
        (
            "source NSFW-RegionNet/bin/activate && python -c 'import ultralytics; print(f\"Ultralytics: {ultralytics.__version__}\")'",
            "Testing Ultralytics installation"
        ),
        (
            "source NSFW-RegionNet/bin/activate && python -c 'import sam2; print(\"SAM2 successfully imported\")'",
            "Testing SAM2 installation"
        ),
        (
            "source NSFW-RegionNet/bin/activate && python -c 'import cv2; print(f\"OpenCV: {cv2.__version__}\")'",
            "Testing OpenCV installation"
        ),
    ]
    
    all_passed = True
    for command, description in test_commands:
        if not run_command(command, description):
            all_passed = False
    
    return all_passed


def main():
    """Main setup function."""
    print("🚀 NSFW-RegionNet Setup Script")
    print("=" * 50)
    
    # Step 1: Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Step 2: Create virtual environment
    if not create_virtual_environment():
        sys.exit(1)
    
    # Step 3: Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies. Please check your internet connection and try again.")
        sys.exit(1)
    
    # Step 4: Download model weights
    if not download_model_weights():
        print("❌ Failed to download model weights. Please check your internet connection.")
        sys.exit(1)
    
    # Step 5: Test installation
    if not test_installation():
        print("⚠️  Some components may not be working correctly. Please check the errors above.")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nQuick Start Commands:")
    print("  source NSFW-RegionNet/bin/activate")
    print("  python chest_exposure_analyzer/main.py --demo")
    print("\nFor more options:")
    print("  python chest_exposure_analyzer/main.py --help")
    print("=" * 50)


if __name__ == "__main__":
    main()