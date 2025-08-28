# NSFW-RegionNet

A computer vision system for content filtering that uses YOLOv11-Pose for human keypoint detection and SAM2 (Segment Anything Model 2) for precise skin region segmentation.

## 🚀 Quick Start

### 1. Automated Setup (Recommended)
```bash
git clone git@github.com:FFFROZEN090/NSFW-RegionNet.git
cd NSFW-RegionNet
python setup.py
```

### 2. Manual Setup
```bash
# Create virtual environment
python -m venv NSFW-RegionNet
source NSFW-RegionNet/bin/activate  # On macOS/Linux
# NSFW-RegionNet\Scripts\activate   # On Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download SAM2 model weights (856MB)
mkdir -p chest_exposure_analyzer/weights
curl -L https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \
  -o chest_exposure_analyzer/weights/sam2_hiera_large.pt
```

### 3. Run Demo
```bash
source NSFW-RegionNet/bin/activate
python chest_exposure_analyzer/main.py --demo
```

## 📋 Usage

### Process Single Image
```bash
python chest_exposure_analyzer/main.py --image path/to/image.jpg
```

### Process Directory of Images
```bash
python chest_exposure_analyzer/main.py --input_dir path/to/images/
```

### Custom Output Directory
```bash
python chest_exposure_analyzer/main.py --demo --output_dir results/
```

## 🏗️ Architecture

### Pipeline Overview
1. **YOLO11-Pose Detection**: Detects human keypoints with 17-point skeleton
2. **Prompt Generation**: Converts facial keypoints to SAM2 prompts (positive facial points, negative background points)
3. **SAM2 Segmentation**: Uses prompts to generate precise skin region masks
4. **Visualization**: Creates comprehensive analysis outputs

### Key Components

#### Models
- **YoloDetector** (`core/models/yolo_detector.py`): YOLOv11-Pose wrapper for human pose detection
- **SamSegmenter** (`core/models/sam2_segmenter.py`): SAM2 wrapper for segmentation with point prompts
- **PromptGenerator** (`core/processors/prompt_generator.py`): Converts keypoints to SAM2 prompts

#### Pipeline
- **ChestExposurePipeline** (`core/pipeline.py`): Orchestrates the complete processing flow
- **VisualizationUtils** (`utils/visualization.py`): Comprehensive visualization tools

## 📊 Output Structure

For each processed image, the system generates:

```
data/output/
└── [image_name]/
    ├── person_1/
    │   ├── keypoints.png          # YOLO keypoint detection
    │   ├── bounding_box.png       # Person bounding box
    │   ├── prompts.png           # SAM2 prompt points (green=+, red=-)
    │   ├── chest_triangle.png    # Chest region triangle
    │   ├── sam2_segmentation.png # SAM2 segmentation result
    │   └── combined.png          # All components combined
    ├── person_2/
    │   └── ...
    └── summary.png               # Multi-person overview
```

## 🔧 Configuration

Edit `chest_exposure_analyzer/configs/default_config.yaml`:

```yaml
# Model paths
models:
  yolo_model_path: "weights/yolov11l-pose.pt"
  sam2_model_path: "chest_exposure_analyzer/weights/sam2_hiera_large.pt"
  sam2_model_type: "hiera_large"

# Detection parameters
detection:
  confidence_threshold: 0.5
  iou_threshold: 0.45
  max_detections: 10

# Segmentation parameters
segmentation:
  multimask_output: true
  
# Visualization settings
visualization:
  keypoint_radius: 5
  line_thickness: 2
  alpha_overlay: 0.5
  save_intermediate_steps: true
```

## 📦 Dependencies

### Core Requirements
- **Python**: 3.8+ (tested with 3.12)
- **PyTorch**: 2.0+ with torchvision
- **OpenCV**: 4.8+ for image processing
- **Ultralytics**: 8.0+ for YOLO11-Pose
- **SAM2**: Facebook's Segment Anything Model 2

### Full Dependencies
- NumPy, Pillow, Matplotlib, scikit-image
- PyYAML, Pydantic for configuration
- Hydra, OmegaConf for SAM2
- pytest, black, flake8 for development

## 🎯 Features

### Robust Detection
- **Multi-person support**: Handles multiple people in a single image
- **Confidence scoring**: Quality assessment for both detection and segmentation
- **Fallback mechanisms**: Graceful degradation when models aren't available

### Advanced Segmentation
- **Prompt-based**: Uses facial keypoints as positive prompts for accurate skin detection
- **Background-aware**: Intelligent negative prompt placement avoids false positives
- **Quality scoring**: SAM2 confidence scores for segmentation assessment

### Comprehensive Visualization
- **Step-by-step outputs**: Every pipeline stage is visualized
- **Multi-person summaries**: Combined visualizations for complex scenes
- **Debug-friendly**: Detailed intermediate results for analysis

## 🔍 Model Information

### SAM2 Model Variants
- `hiera_large` (default): 856MB, highest accuracy
- `hiera_base_plus`: Smaller, faster alternative
- `hiera_small`: Lightweight version
- `hiera_tiny`: Minimal resource usage

### YOLO11-Pose
- Automatically downloads YOLOv11n-pose if custom weights not found
- Supports custom trained models via `yolo_model_path` config

## 🚨 Important Notes

### Content Filtering Purpose
This system is designed for **defensive security** and **content moderation** purposes. It helps platforms:
- Detect potentially inappropriate content
- Provide automated content filtering
- Support human moderators with analysis tools

### Privacy and Ethics
- **No data storage**: Processes images locally without external transmission
- **Configurable sensitivity**: Adjustable thresholds for different use cases
- **Transparent processing**: All intermediate steps are visualizable

## 🛠️ Development

### Running Tests
```bash
source NSFW-RegionNet/bin/activate
pytest chest_exposure_analyzer/tests/
```

### Code Formatting
```bash
black chest_exposure_analyzer/
flake8 chest_exposure_analyzer/
```

### Adding New Features
The modular architecture makes it easy to:
- Replace detection models (implement `YoloDetector` interface)
- Add segmentation models (implement `SamSegmenter` interface)
- Create custom prompt generators (extend `PromptGenerator`)
- Add new visualizations (extend `VisualizationUtils`)

## 📈 Performance

### Typical Processing Times (CPU)
- **Single person image**: 3-5 seconds
- **Multi-person image**: 5-10 seconds
- **Batch processing**: ~4 seconds per image average

### Resource Usage
- **Memory**: ~2-4GB RAM for large images
- **Storage**: ~856MB for SAM2 model weights
- **GPU**: Optional but recommended for faster processing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run quality checks: `black . && flake8 . && pytest`
5. Submit a pull request

## 📄 License

This project is intended for defensive security and content moderation purposes. Please review the licensing terms before use.

## 🆘 Troubleshooting

### Common Issues

**SAM2 not loading:**
```bash
# Verify SAM2 installation
source NSFW-RegionNet/bin/activate
python -c "import sam2; print('SAM2 OK')"

# Re-download weights if corrupted
rm chest_exposure_analyzer/weights/sam2_hiera_large.pt
curl -L https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \
  -o chest_exposure_analyzer/weights/sam2_hiera_large.pt
```

**Memory issues:**
- Use smaller SAM2 model variant in config
- Process images individually instead of batch mode
- Resize large images before processing

**Dependencies conflicts:**
```bash
# Clean reinstall
rm -rf NSFW-RegionNet/
python setup.py
```

## 📞 Support

For issues related to:
- **SAM2**: [Facebook SAM2 Repository](https://github.com/facebookresearch/segment-anything-2)
- **YOLO11**: [Ultralytics Documentation](https://docs.ultralytics.com/)
- **This project**: Create an issue in this repository

---

**⚡ Quick Test:**
```bash
source NSFW-RegionNet/bin/activate && python chest_exposure_analyzer/main.py --demo
```