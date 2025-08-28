# NSFW-RegionNet

基于计算机视觉的内容过滤系统，使用 YOLOv11-Pose 进行人体关键点检测，SAM2 进行精确皮肤区域分割，并实现胸部暴露检测和自动马赛克处理。

## 安装配置

### 自动安装（推荐）
```bash
git clone git@github.com:FFFROZEN090/NSFW-RegionNet.git
cd NSFW-RegionNet
python setup.py
```

### 手动安装
```bash
# 创建虚拟环境
python -m venv NSFW-RegionNet
source NSFW-RegionNet/bin/activate  # macOS/Linux
# NSFW-RegionNet\Scripts\activate   # Windows

# 安装依赖
pip install --upgrade pip
pip install -r requirements.txt

# 下载 SAM2 模型权重 (856MB)
mkdir -p chest_exposure_analyzer/weights
curl -L https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \
  -o chest_exposure_analyzer/weights/sam2_hiera_large.pt
```

## 核心使用方法

### 1. 演示模式
```bash
source NSFW-RegionNet/bin/activate
python chest_exposure_analyzer/main.py --demo
```

### 2. 单张图片处理
```bash
python chest_exposure_analyzer/main.py --image path/to/image.jpg
```

### 3. 生产部署模式
```bash
python chest_exposure_analyzer/main.py --deploy input_folder/ output_folder/
```
此模式自动输出：
- 有暴露内容的图片：自动打马赛克版本
- 正常图片：保持原始版本
- 无调试文件，仅输出最终结果

### 4. Jupyter 交互演示
```bash
jupyter notebook demo_pipeline.ipynb
```

## 处理流程

1. **姿态检测**：YOLOv11-Pose 检测人体 17 个关键点
2. **提示生成**：将关键点转换为 SAM2 提示点
3. **皮肤分割**：SAM2 基于提示点分割皮肤区域
4. **胸部区域分析**：生成解剖学正确的胸部三角形区域
5. **暴露检测**：分析皮肤与胸部区域交集
6. **形态学处理**：开运算和闭运算优化检测区域
7. **马赛克处理**：对检测到的暴露区域应用马赛克

## 程序化接口

### 基础处理
```python
from chest_exposure_analyzer.core.pipeline import ChestExposurePipeline

# 初始化管道
pipeline = ChestExposurePipeline()

# 处理单张图片
results = pipeline.process_image("path/to/image.jpg")

# 批量处理
results = pipeline.process_batch("input_dir", "output_dir")
```

### 生产部署接口
```python
# 自动处理和马赛克
stats = pipeline.process_for_deployment("input_dir", "output_dir")

print(f"处理总数: {stats['processed']}")
print(f"打码图片: {stats['exposed']}")
print(f"原始图片: {stats['processed'] - stats['exposed']}")
```

### 暴露分析
```python
from chest_exposure_analyzer.core.processors.chest_analyzer import ChestExposureAnalyzer

analyzer = ChestExposureAnalyzer(
    min_intersection_ratio=0.01,
    min_intersection_area=100,
    mosaic_block_size=30
)

# 分析暴露情况
analysis = analyzer.analyze_chest_exposure(skin_mask, chest_mask, detection)
print(f"是否暴露: {analysis['is_exposed']}")
print(f"置信度: {analysis['analysis_confidence']}")

# 应用马赛克
mosaicked_image = analyzer.apply_mosaic_to_regions(image, mask3_refined)
```

## 配置参数

编辑 `chest_exposure_analyzer/configs/default_config.yaml`：

```yaml
# 模型路径
models:
  yolo_model_path: "weights/yolov11l-pose.pt"
  sam2_model_path: "chest_exposure_analyzer/weights/sam2_hiera_large.pt"

# 检测参数
detection:
  confidence_threshold: 0.5

# 暴露检测参数
exposure_detection:
  min_intersection_ratio: 0.01    # 最小交集比例
  min_intersection_area: 100      # 最小交集面积（像素）
  min_confidence_threshold: 0.5   # 最小置信度阈值
  
  # 形态学操作参数
  morphology_kernel_size: 5       # 形态学操作核大小
  opening_iterations: 1           # 开运算迭代次数
  closing_iterations: 2           # 闭运算迭代次数
  
  # 马赛克参数
  mosaic_block_size: 30          # 马赛克块大小
  mosaic_intensity: 1.0          # 马赛克强度
```

## 输出结构

```
data/output/
└── [图片名称]/
    ├── person_1/
    │   ├── keypoints.png              # 关键点检测
    │   ├── prompts.png               # SAM2 提示点
    │   ├── chest_triangle.png        # 胸部三角区域
    │   ├── sam2_segmentation.png     # 皮肤分割结果
    │   ├── exposure_analysis.png     # 暴露分析
    │   ├── morphology_comparison.png # 形态学处理对比
    │   └── mosaic_comparison.png     # 马赛克处理对比
    └── summary.png                   # 多人概览
```

## 主要组件

### 核心模块
- **ChestExposurePipeline**：主处理管道
- **YoloDetector**：YOLO 姿态检测器
- **SamSegmenter**：SAM2 分割器
- **PromptGenerator**：提示点生成器
- **ChestExposureAnalyzer**：暴露检测分析器

### 关键功能
- **多人支持**：单张图片处理多个人物
- **置信度评分**：检测和分割质量评估
- **自动马赛克**：检测到暴露内容自动处理
- **形态学优化**：开闭运算优化检测区域
- **可视化调试**：每个处理步骤可视化

## 性能指标

### 处理时间（CPU）
- 单人图片：7-8 秒
- 多人图片：15-20 秒

### 资源占用
- 内存：2-4GB RAM
- 存储：856MB 模型权重
- GPU：可选，建议使用以加速处理

## 算法参数说明

### 暴露检测阈值
- `min_intersection_ratio`：皮肤与胸部区域交集比例阈值
- `min_intersection_area`：交集面积像素阈值
- `min_confidence_threshold`：整体检测置信度阈值

### 形态学操作
- `morphology_kernel_size`：形态学操作核大小，影响噪点去除效果
- `opening_iterations`：开运算次数，去除小噪点
- `closing_iterations`：闭运算次数，填补空洞

### 马赛克处理
- `mosaic_block_size`：马赛克块大小，数值越大越模糊
- `mosaic_intensity`：马赛克强度，1.0 为完全马赛克，0.0 为无效果

## 故障排除

### SAM2 加载失败
```bash
# 验证 SAM2 安装
python -c "import sam2; print('SAM2 OK')"

# 重新下载权重文件
rm chest_exposure_analyzer/weights/sam2_hiera_large.pt
curl -L https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \
  -o chest_exposure_analyzer/weights/sam2_hiera_large.pt
```

### 内存不足
- 使用较小的 SAM2 模型
- 逐张处理图片而非批量处理
- 预先压缩大尺寸图片

### 依赖冲突
```bash
# 清理重装
rm -rf NSFW-RegionNet/
python setup.py
```

## 开发测试

```bash
# 运行测试
pytest chest_exposure_analyzer/tests/

# 代码格式化
black chest_exposure_analyzer/
flake8 chest_exposure_analyzer/
```

本系统专为内容过滤和平台安全而设计，提供本地化处理，无数据外传，支持灵活的阈值配置以适应不同使用场景。
