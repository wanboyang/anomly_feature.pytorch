"""
Dataset creator for C3D feature combination / C3D特征组合的数据集创建器
This module processes C3D features for video anomaly detection datasets.
此模块为视频异常检测数据集处理C3D特征。
"""

import numpy as np
import os
import glob
from multiprocessing import Pool
from tqdm import tqdm
import json
import pandas as pd
import pickle
import argparse

def combine_feature(video):
    """
    Combine C3D features for a single video / 为单个视频组合C3D特征
    
    This function loads C3D features for a video from different layers and
    combines them into a single feature file per video.
    此函数从不同层加载视频的C3D特征，并将它们组合成每个视频的单个特征文件。
    
    Args:
        video: Name of the video / 视频名称
    
    Returns:
        None
    """
    layers = ['pool5']
    for layer in layers:
        rgb_fea = []
        # Load and sort RGB features for C3D model
        # 加载并排序C3D模型的RGB特征
        rgb_features = glob.glob(os.path.join('/home/tu-wan/windowswan/dataset/{}//features//{}'.format(dataset, pretrain_model), layer, 'rgb',video, '*.npy'))
        rgb_features.sort(key=lambda x: int(x.split('_')[-1].split('.')[0].zfill(8)))
        
        # Load features for each frame
        # 为每个帧加载特征
        for i in range(len(rgb_features)):
            rgb_fea_np = np.load(rgb_features[i])
            rgb_fea.append(rgb_fea_np)
        
        # Convert to numpy array
        # 转换为numpy数组
        rgb_fea = np.asarray(rgb_fea)
        
        # Create save directory if it doesn't exist
        # 如果保存目录不存在则创建
        if os.path.exists(os.path.join('/home/tu-wan/windowswan/dataset/{}//features_video//{}'.format(dataset, pretrain_model), layer, 'rgb', video)) == 0:
            os.makedirs(os.path.join('/home/tu-wan/windowswan/dataset/{}//features_video//{}'.format(dataset, pretrain_model), layer, 'rgb', video))
        
        # Save combined features
        # 保存组合特征
        np.save(file=os.path.join('/home/tu-wan/windowswan/dataset/{}//features_video//{}'.format(dataset, pretrain_model), layer, 'rgb', video, 'feature.npy'), arr=rgb_fea)




if __name__ == "__main__":
    """
    Main entry point for C3D dataset creation / C3D数据集创建的主入口点
    
    This script processes all videos in parallel to combine C3D features
    for video anomaly detection datasets.
    此脚本并行处理所有视频，为视频异常检测数据集组合C3D特征。
    """
    
    # Parse command line arguments
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help='Name of dataset', default='shanghaitech', type=str)
    args = parser.parse_args()
    
    # Set dataset and model parameters for C3D
    # 为C3D设置数据集和模型参数
    pretrain_model = 'c3d'
    dataset = args.dataset
    feature_dir = '/home/tu-wan/windowswan/dataset/{}/features/{}'.format(dataset, pretrain_model)
    
    # Use pool5 layer for C3D feature combination
    # 使用pool5层进行C3D特征组合
    layer = 'pool5'
    
    # Combine RGB features for every video (C3D typically uses RGB only)
    # 为每个视频组合RGB特征（C3D通常仅使用RGB）
    rgb_feature_dir = os.path.join(feature_dir, layer, 'rgb')
    videos = os.listdir(rgb_feature_dir)
    
    # Process videos in parallel using multiprocessing
    # 使用多处理并行处理视频
    with Pool(processes=6) as p:
        max_ = len(videos)
        with tqdm(total=max_) as pbar:
            for i, _ in tqdm(enumerate(p.imap_unordered(combine_feature, videos))):
                pbar.update()
