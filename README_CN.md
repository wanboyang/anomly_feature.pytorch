# 异常特征提取：用于视频分析的C3D和I3D特征提取工具

[![许可证](https://img.shields.io/badge/许可证-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.5%2B-yellow)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.2.0-orange)](https://pytorch.org/)

> **语言**: [English](README.md) | [中文](README_CN.md)

---

## 摘要

本仓库提供了一个全面的工具包，用于从视频中提取 **C3D** 和 **I3D** 特征，专门为视频异常检测和分析任务设计。该框架支持多个数据集和特征模态，使研究人员能够高效地提取时空特征，用于各种视频理解应用。

## 🎯 主要特性

- **多模型支持**: 使用C3D和I3D架构提取特征
- **多模态特征**: 支持RGB和光流模态
- **数据集兼容性**: 适用于ShanghaiTech、UCF-Crime和其他视频数据集
- **高效处理**: 支持多进程视频帧提取
- **灵活配置**: 可自定义特征提取层和参数

## 🏗️ 支持的模型

### C3D (卷积3D)
- **架构**: 用于时空特征学习的3D卷积网络
- **特征**: 从fc6、fc7或pool5层提取特征
- **应用**: 通用视频理解和动作识别

### I3D (膨胀3D卷积网络)
- **架构**: 将2D CNN膨胀为3D用于视频分析
- **模态**: RGB和光流流
- **特征**: 在Kinetics数据集上预训练
- **应用**: 最先进的视频分类和异常检测

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/wanboyang/anomly_feature.pytorch.git
cd anomly_feature.pytorch

# 创建环境
conda env create -f anomaly_feature.yaml
conda activate anomaly_icme
```

### 模型下载

从[百度网盘](https://pan.baidu.com/s/1g4XGLqRoRJhQwIGtHif2jg) (密码: dzqm)下载预训练模型并解压到根目录。

### 视频预处理

从视频中提取帧：
```bash
python Videos_to_frames_multiprocessing.py
```

**推荐**: 对于I3D特征提取，使用 [denseflow](https://github.com/open-mmlab/denseflow) 生成RGB和光流图像。

### 输入列表生成

生成特征提取的索引文件：
```bash
python ./dataset/write_data_label_txt_new.py
```

## 🔧 特征提取

### I3D RGB特征
```bash
python feature_extract.py --dataset shanghaitech --modelName i3d --snapshot ./model/i3d/i3d_model_weight/model_kinetics_rgb.pth --datamodal rgb
```

### I3D 光流特征
```bash
python feature_extract.py --dataset shanghaitech --modelName i3d --snapshot ./model/i3d/i3d_model_weight/model_kinetics_flow.pth --datamodal flow
```

### C3D 特征
```bash
python feature_extract.py --dataset shanghaitech --modelName c3d --snapshot ./model/c3d/c3d.pickle --datamodal rgb
```

### 附加选项
- `--fc_layer`: 指定特征提取层（C3D的fc6、fc7、pool5）
- `--dataset`: 选择数据集（shanghaitech等）

## 📊 数据集支持

### 当前支持：
- **ShanghaiTech**: 大规模视频异常检测数据集
- **UCF-Crime**: 真实世界监控视频数据集
- **自定义数据集**: 可轻松扩展到其他视频数据集

### 数据集结构：
```
dataset/
├── {数据集名称}/
│   ├── {模型名称}/
│   │   ├── {模态}_list.txt
│   │   └── label.txt
│   └── features/
│       └── {模型名称}/
│           └── {模态}/
```

## 🛠️ 附加工具

### 数据集创建
生成最终特征文件：
```bash
python dataset_creater.py
```

### C3D特定数据集创建
```bash
python dataset_creater_C3D.py
```

### 视频分割
```bash
python clip2segment.py
```

## 📈 性能特点

- **高效处理**: 多线程特征提取
- **内存优化**: 大数据集的批处理
- **高质量特征**: 最先进的预训练模型
- **兼容性**: 与主要视频分析框架兼容

## 🤝 集成

此特征提取工具包设计用于与以下框架无缝协作：
- [anomaly_detection_LAD2000](https://github.com/wanboyang/anomaly_detection_LAD2000)
- [Anomaly_AR_Net_ICME_2020](https://github.com/wanboyang/Anomaly_AR_Net_ICME_2020)
- 其他视频异常检测框架

## 📧 联系方式

如有问题和建议，请联系：
- **万博洋** - wanboyangjerry@163.com
