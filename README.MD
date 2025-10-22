# Anomaly Feature Extraction: C3D and I3D Feature Extraction for Video Analysis

[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.5%2B-yellow)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.2.0-orange)](https://pytorch.org/)

> **Language**: [English](README.md) | [中文](README_CN.md)

---

## Abstract

This repository provides a comprehensive toolkit for extracting **C3D** and **I3D** features from videos, specifically designed for video anomaly detection and analysis tasks. The framework supports multiple datasets and feature modalities, enabling researchers to efficiently extract spatio-temporal features for various video understanding applications.

## 🎯 Key Features

- **Multi-model Support**: Extract features using both C3D and I3D architectures
- **Multi-modal Features**: Support for RGB and optical flow modalities
- **Dataset Compatibility**: Works with ShanghaiTech, UCF-Crime, and other video datasets
- **Efficient Processing**: Multi-processing support for video frame extraction
- **Flexible Configuration**: Customizable feature extraction layers and parameters

## 🏗️ Supported Models

### C3D (Convolutional 3D)
- **Architecture**: 3D convolutional networks for spatio-temporal feature learning
- **Features**: Extracts features from fc6, fc7, or pool5 layers
- **Applications**: General video understanding and action recognition

### I3D (Inflated 3D ConvNet)
- **Architecture**: 2D CNN inflated to 3D for video analysis
- **Modalities**: RGB and optical flow streams
- **Features**: Pre-trained on Kinetics dataset
- **Applications**: State-of-the-art video classification and anomaly detection

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/wanboyang/anomly_feature.pytorch.git
cd anomly_feature.pytorch

# Create environment
conda env create -f anomaly_feature.yaml
conda activate anomaly_icme
```

### Model Download

Download pre-trained models from [Baidu Netdisk](https://pan.baidu.com/s/1g4XGLqRoRJhQwIGtHif2jg) (password: dzqm) and extract to the root directory.

### Video Preprocessing

Extract frames from videos:
```bash
python Videos_to_frames_multiprocessing.py
```

**Recommendation**: For I3D feature extraction, use [denseflow](https://github.com/open-mmlab/denseflow) to generate RGB and optical flow images.

### Input List Generation

Generate index files for feature extraction:
```bash
python ./dataset/write_data_label_txt_new.py
```

## 🔧 Feature Extraction

### I3D RGB Features
```bash
python feature_extract.py --dataset shanghaitech --modelName i3d --snapshot ./model/i3d/i3d_model_weight/model_kinetics_rgb.pth --datamodal rgb
```

### I3D Flow Features
```bash
python feature_extract.py --dataset shanghaitech --modelName i3d --snapshot ./model/i3d/i3d_model_weight/model_kinetics_flow.pth --datamodal flow
```

### C3D Features
```bash
python feature_extract.py --dataset shanghaitech --modelName c3d --snapshot ./model/c3d/c3d.pickle --datamodal rgb
```

### Additional Options
- `--fc_layer`: Specify feature extraction layer (fc6, fc7, pool5 for C3D)
- `--dataset`: Choose dataset (shanghaitech, etc.)

## 📊 Dataset Support

### Currently Supported:
- **ShanghaiTech**: Large-scale video anomaly detection dataset
- **UCF-Crime**: Real-world surveillance video dataset
- **Custom Datasets**: Easily extendable to other video datasets

### Dataset Structure:
```
dataset/
├── {dataset_name}/
│   ├── {model_name}/
│   │   ├── {datamodal}_list.txt
│   │   └── label.txt
│   └── features/
│       └── {model_name}/
│           └── {datamodal}/
```

## 🛠️ Additional Tools

### Dataset Creation
Generate final feature files:
```bash
python dataset_creater.py
```

### C3D-specific Dataset Creation
```bash
python dataset_creater_C3D.py
```

### Video Segmentation
```bash
python clip2segment.py
```

## 📈 Performance

- **Efficient Processing**: Multi-threaded feature extraction
- **Memory Optimized**: Batch processing for large datasets
- **Quality Features**: State-of-the-art pre-trained models
- **Compatible**: Features compatible with major video analysis frameworks

## 🤝 Integration

This feature extraction toolkit is designed to work seamlessly with:
- [anomaly_detection_LAD2000](https://github.com/wanboyang/anomaly_detection_LAD2000)
- [Anomaly_AR_Net_ICME_2020](https://github.com/wanboyang/Anomaly_AR_Net_ICME_2020)
- Other video anomaly detection frameworks

## 📝 Code Documentation

All Python files in this repository have been comprehensively annotated with both English and Chinese comments to improve code readability and maintainability. The annotations include:

- **Module-level documentation** explaining the purpose and functionality of each file
- **Function-level documentation** with detailed parameter descriptions and return values
- **Inline comments** explaining complex logic and implementation details
- **Bilingual support** with both English and Chinese explanations

### Example of Annotated Code Structure:
```python
"""
Feature extraction module for video anomaly detection / 视频异常检测的特征提取模块
This module extracts spatio-temporal features from videos using pre-trained models (I3D, C3D).
此模块使用预训练模型（I3D、C3D）从视频中提取时空特征。
"""

def feature(data_path, dataset, snapshot, modelName, dataloader, datamodal='rgb', fc_layer='fc6'):
    """
    Main feature extraction function / 主要特征提取函数
    
    This function initializes the model, loads pre-trained weights, and extracts features
    from video clips using the specified model architecture.
    此函数初始化模型，加载预训练权重，并使用指定的模型架构从视频片段中提取特征。
    
    Args:
        data_path: Root path for data storage / 数据存储的根路径
        dataset: Name of the dataset / 数据集名称
        snapshot: Path to pre-trained model weights / 预训练模型权重路径
        modelName: Model architecture ('c3d' or 'i3d') / 模型架构 ('c3d' 或 'i3d')
        dataloader: DataLoader for video clips / 视频片段的数据加载器
        datamodal: Data modality ('rgb' or 'flow') / 数据模态 ('rgb' 或 'flow')
        fc_layer: Feature extraction layer for C3D model / C3D模型的特征提取层
    
    Returns:
        None
    """
```

## 📧 Contact

For questions and suggestions, please contact:
- **Boyang Wan** - wanboyangjerry@163.com
