# API Usage Guide

## 简单API调用

### 基本用法

```python
from chest_exposure_analyzer.api import process_image_for_content_filtering

# 处理单张图片
result = process_image_for_content_filtering("/path/to/image.jpg")

# 检查结果
if result['needs_mosaic']:
    print(f"图片需要打码，已保存到: {result['output_path']}")
else:
    print(f"图片无需打码，已保存到: {result['output_path']}")
```

### 输出说明

函数返回字典包含：
- `needs_mosaic` (bool): 是否需要打码
- `original_path` (str): 原图路径
- `output_path` (str): 输出图片路径
- `status` (str): "ORIGINAL" 或 "MOSAICKED"
- `person_count` (int): 检测到的人物总数
- `frontal_person_count` (int): 正面朝向的人物数量

### 文件命名规则

输出图片会自动在文件名中标注处理状态：
- `image_ORIGINAL.jpg` - 无需打码的原图
- `image_MOSAICKED.jpg` - 需要打码的处理后图片

## 命令行使用

```bash
# 使用示例脚本
python example_usage.py /path/to/image.jpg

# 或直接使用main.py
python chest_exposure_analyzer/main.py --image /path/to/image.jpg
```

## 处理逻辑

1. **人物检测**: 使用YOLOv11-Pose检测图片中的所有人物
2. **面部朝向过滤**: 只处理正面朝向的人物（3区域检测：鼻子+左右眼/耳）
3. **皮肤分割**: 使用SAM2对正面人物进行皮肤区域分割
4. **胸部分析**: 检测胸部三角区域与皮肤区域的交集
5. **智能打码**: 对检测到的暴露区域进行马赛克处理
6. **文件输出**: 根据检测结果保存原图或打码后的图片

## 批量处理

```python
from chest_exposure_analyzer.api import create_content_filter_pipeline

pipeline = create_content_filter_pipeline()

images = ["/path/to/image1.jpg", "/path/to/image2.jpg"]
for image_path in images:
    result = pipeline.process_single_image_file(image_path)
    print(f"{image_path}: {result['status']}")
```