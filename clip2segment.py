"""
Clip to segment feature processing / 片段到段落的特征处理
This module converts clip-level features to segment-level features using adaptive pooling.
此模块使用自适应池化将片段级特征转换为段落级特征。
"""

import torch
import torch.nn.functional as f
import os
import glob
import numpy as np
from tqdm import tqdm

# Load all C3D clip features
# 加载所有C3D片段特征
clip_features = glob.glob('/home/tu-wan/windowswan/dataset/UCSDPed2/features_video/c3d/fc6/rgb/*/*')

# Process each clip feature
# 处理每个片段特征
for clip_feature in tqdm(clip_features):
    # Load and convert feature to tensor
    # 加载特征并转换为张量
    tmp_f = np.load(clip_feature)
    tmp_f = torch.from_numpy(tmp_f).unsqueeze(0)
    feature_d = tmp_f.shape[-1]
    
    # Apply adaptive average pooling to convert to segment-level features
    # 应用自适应平均池化转换为段落级特征
    tar_f = f.adaptive_avg_pool2d(tmp_f, (32,feature_d))
    tar_f = tar_f.squeeze().numpy()
    
    # Create save directory if it doesn't exist
    # 如果保存目录不存在则创建
    if os.path.exists(os.path.split(clip_feature.replace('c3d', 'c3d_segments'))[0]) == 0:
        os.makedirs(os.path.split(clip_feature.replace('c3d', 'c3d_segments'))[0])
    
    # Save processed segment features
    # 保存处理后的段落特征
    np.save(file=clip_feature.replace('c3d', 'c3d_segments'),arr=tar_f)
