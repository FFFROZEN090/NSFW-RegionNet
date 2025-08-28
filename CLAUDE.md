# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NSFW-RegionNet is a content filtering system for platforms that uses computer vision to detect and analyze potentially inappropriate chest exposure in images. The system employs a two-stage pipeline:

1. **Pose Detection**: Uses YOLOv11-Pose to detect human keypoints
2. **Region Analysis**: Uses Segment Anything Model 2 (SAM2) with keypoint prompts to segment skin regions and analyze chest areas

## Development Environment

### Virtual Environment Setup
```bash
# Activate the project virtual environment
source NSFW-RegionNet/bin/activate  # macOS/Linux
# NSFW-RegionNet\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Git Configuration
Remote repository: `git@github.com:FFFROZEN090/NSFW-RegionNet.git`

## Project Structure

```
chest_exposure_analyzer/
├── main.py                          # Main entry point
├── core/
│   ├── pipeline.py                  # Core processing orchestrator
│   ├── data_models.py              # Data structures and contracts
│   ├── models/
│   │   ├── yolo_detector.py        # YOLOv11-Pose wrapper
│   │   └── sam2_segmenter.py       # SAM2 model wrapper
│   └── processors/
│       ├── prompt_generator.py     # Keypoint to SAM2 prompt conversion
│       └── chest_analyzer.py       # Chest region analysis logic
├── utils/
│   ├── visualization.py           # Visualization utilities
│   └── image_utils.py             # Image processing utilities
├── configs/
│   └── default_config.yaml        # Configuration parameters
├── weights/                        # Model weight files (.pt, .pth)
├── data/
│   ├── input/                      # Input images
│   └── output/                     # Results and visualizations
└── tests/                          # Unit and integration tests
```

## Common Development Commands

```bash
# Run the main pipeline
python chest_exposure_analyzer/main.py

# Run tests
pytest chest_exposure_analyzer/tests/

# Code formatting
black chest_exposure_analyzer/

# Linting
flake8 chest_exposure_analyzer/

# Install in development mode
pip install -e .
```

## Key Dependencies

- **torch/torchvision**: PyTorch framework for deep learning
- **ultralytics**: YOLOv11-Pose implementation
- **SAM-2**: Meta's Segment Anything Model 2
- **opencv-python**: Computer vision operations
- **scikit-image**: Image processing utilities

## Model Weights

Required model files should be placed in `weights/`:
- `yolov11l-pose.pt`: YOLOv11 pose detection model
- `sam2_hiera_large.pt`: SAM2 Hiera Large checkpoint

Download instructions will be added once implementation begins.

## Configuration

All parameters are centralized in `configs/default_config.yaml`:
- Model paths and parameters
- Detection thresholds
- Segmentation settings
- Visualization options
- I/O paths

## Architecture Principles

- **High cohesion, low coupling**: Each module has focused responsibilities
- **Data contracts**: Well-defined interfaces between components
- **Configurable**: Parameters externalized to YAML configuration
- **Debuggable**: Intermediate visualization outputs for pipeline inspection
- **Testable**: Modular design supports unit testing

## Development Notes

- This is a content filtering system for platform safety
- Focus on accuracy and reliability over speed
- Extensive visualization support for debugging and validation
- Modular architecture allows for easy component replacement/improvement