"""
Dataset creator for video feature combination / 视频特征组合的数据集创建器
This module combines RGB and optical flow features for video anomaly detection datasets.
此模块为视频异常检测数据集组合RGB和光流特征。
"""

import numpy as np
import os
import glob
from multiprocessing import Pool
from tqdm import tqdm
import argparse

def combine_feature(video):
    """
    Combine RGB and flow features for a single video / 为单个视频组合RGB和光流特征
    
    This function loads RGB and optical flow features for a video, combines them,
    and saves the combined features along with individual modality features.
    此函数加载视频的RGB和光流特征，组合它们，并保存组合特征以及各个模态的特征。
    
    Args:
        video: Name of the video / 视频名称
    
    Returns:
        None
    """
    combine_fea = []
    rgb_fea = []
    flow_fea = []
    
    # Load and sort RGB features
    # 加载并排序RGB特征
    rgb_features = glob.glob(os.path.join('/home/tu-wan/windowswan/dataset/{}//features//{}'.format(dataset, pretrain_model),'rgb',video, '*.npy'))
    rgb_features.sort(key=lambda x: int(x[-9:-4]))
    
    # Load and sort flow features
    # 加载并排序光流特征
    flow_features = glob.glob(os.path.join('/home/tu-wan/windowswan/dataset/{}//features//{}'.format(dataset, pretrain_model), 'flow', video, '*.npy'))
    flow_features.sort(key=lambda x: int(x[-9:-4]))
    
    # Combine features frame by frame
    # 逐帧组合特征
    for i in range(len(rgb_features)):
        rgb_fea_np = np.load(rgb_features[i])
        flow_fea_np = np.load(flow_features[i])
        rgb_fea.append(rgb_fea_np)
        flow_fea.append(flow_fea_np)
        feature = np.hstack((rgb_fea_np,flow_fea_np))
        combine_fea.append(feature)
    
    # Convert lists to numpy arrays
    # 将列表转换为numpy数组
    combine_fea = np.asarray(combine_fea)
    rgb_fea = np.asarray(rgb_fea)
    flow_fea = np.asarray(flow_fea)
    
    # Create save directories
    # 创建保存目录
    save_path = os.path.join('/home/tu-wan/windowswan/dataset/{}//features_video//{}'.format(dataset, pretrain_model), 'combine_flownet',video)
    if os.path.exists(save_path) == 0:
        os.makedirs(save_path)
    if os.path.exists(os.path.join('/home/tu-wan/windowswan/dataset/{}//features_video//{}'.format(dataset, pretrain_model), 'rgb', video)) == 0:
        os.makedirs(os.path.join('/home/tu-wan/windowswan/dataset/{}//features_video//{}'.format(dataset, pretrain_model), 'rgb', video))
    if os.path.exists(os.path.join('/home/tu-wan/windowswan/dataset/{}//features_video//{}'.format(dataset, pretrain_model), 'flownet', video)) == 0:
        os.makedirs(os.path.join('/home/tu-wan/windowswan/dataset/{}//features_video//{}'.format(dataset, pretrain_model), 'flownet', video))
    
    # Save combined and individual features
    # 保存组合和单独的特征
    np.save(file=os.path.join(save_path, 'feature.npy'),arr=combine_fea)
    np.save(file=os.path.join('/home/tu-wan/windowswan/dataset/{}//features_video//{}'.format(dataset, pretrain_model), 'rgb', video, 'feature.npy'), arr=rgb_fea)
    np.save(file=os.path.join('/home/tu-wan/windowswan/dataset/{}//features_video//{}'.format(dataset, pretrain_model), 'flownet', video,  'feature.npy'), arr=flow_fea)



if __name__ == "__main__":
    """
    Main entry point for dataset creation / 数据集创建的主入口点
    
    This script processes all videos in parallel to combine RGB and flow features
    for video anomaly detection datasets.
    此脚本并行处理所有视频，为视频异常检测数据集组合RGB和光流特征。
    """
    
    # Parse command line arguments
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help='Name of dataset', default='shanghaitech', type=str)
    args = parser.parse_args()
    
    # Set dataset and model parameters
    # 设置数据集和模型参数
    pretrain_model = 'i3d'
    dataset = args.dataset
    feature_dir = '/home/tu-wan/windowswan/dataset/{}/features/{}'.format(dataset, pretrain_model)
    
    # Combine RGB and flow features for every video
    # 为每个视频组合RGB和光流特征
    rgb_feature_dir = os.path.join(feature_dir, 'rgb')
    flow_feature_dir = os.path.join(feature_dir, 'flow')
    videos = os.listdir(rgb_feature_dir)
    
    # Process videos in parallel using multiprocessing
    # 使用多处理并行处理视频
    with Pool(processes=6) as p:
        max_ = len(videos)
        with tqdm(total=max_) as pbar:
            for i, _ in tqdm(enumerate(p.imap_unordered(combine_feature, videos))):
                pbar.update()
