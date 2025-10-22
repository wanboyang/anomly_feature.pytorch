"""
Feature extraction module for video anomaly detection / 视频异常检测的特征提取模块
This module extracts spatio-temporal features from videos using pre-trained models (I3D, C3D).
此模块使用预训练模型（I3D、C3D）从视频中提取时空特征。
"""

import timeit
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
from dataset.Datasetloader import trainDataset, txttans
from model.i3d.src.i3dpt import I3D
from model.C3D_model import C3D
import sys



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
    ###################### Build model #####################################
    train_params = []
    if modelName == 'c3d':
        model = C3D(nb_classes=487)
        train_params.append({'params': model.parameters()})
    elif modelName == 'i3d':
        if datamodal == 'rgb':
            model = I3D(400, modality='rgb', dropout_prob=0, name='inception')
        else:
            model = I3D(400, modality='flow', dropout_prob=0, name='inception')
    else:
        print('We only implemented C3D or i3d models.')
        raise NotImplementedError
    
    ###################### Load weights #####################################
    if snapshot:
        model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(snapshot).items()})
    
    # Set feature extraction layer and save directory based on model type
    # 根据模型类型设置特征提取层和保存目录
    if modelName == 'c3d':
        if fc_layer == 'fc6':
            feature_layer = 6
        elif fc_layer == 'fc7':
            feature_layer = 7
        elif fc_layer == 'pool5':
            feature_layer = 5
        feature_save_dir = os.path.join(data_path, 'dataset', dataset, 'features', modelName, fc_layer, datamodal)
    elif modelName == 'i3d':
        feature_layer = 5
        feature_save_dir = os.path.join(data_path, 'dataset', dataset, 'features', modelName, datamodal)

    model.to(device)

    # Create feature save directory if it doesn't exist
    # 如果特征保存目录不存在则创建
    if os.path.exists(feature_save_dir) == 0:
        os.makedirs(feature_save_dir)
    
    ###################### Log feature extraction info ######################
    if os.path.exists(os.path.join('./model_feature/', dataset, modelName)) == 0:
        os.makedirs(os.path.join('./model_feature/', dataset, modelName))
    with open(file=os.path.join('./model_feature/', dataset, modelName,'feature.txt'), mode='a+') as f:
        f.write("dataset:{} ".format(dataset)+ '\n')
        f.write("snapshot:{} ".format(snapshot) + '\n')
        f.write("savedir:{} ".format(feature_save_dir) + '\n')
        f.write("========================================== " + '\n')

    # Perform feature extraction
    # 执行特征提取
    model_feature(model=model,dataloader=dataloader, feature_save_dir=feature_save_dir,datamodal=datamodal,dataset=dataset, feature_layer=feature_layer)



def model_feature(model, dataloader, feature_save_dir, datamodal, dataset, feature_layer=None):
    """
    Extract features from video clips and save to files / 从视频片段中提取特征并保存到文件
    
    This function processes video clips through the model, extracts features, and saves
    them as numpy files organized by video name and frame range.
    此函数通过模型处理视频片段，提取特征，并按视频名称和帧范围组织保存为numpy文件。
    
    Args:
        model: Pre-trained model for feature extraction / 用于特征提取的预训练模型
        dataloader: DataLoader containing video clips / 包含视频片段的数据加载器
        feature_save_dir: Directory to save extracted features / 保存提取特征的目录
        datamodal: Data modality ('rgb' or 'flow') / 数据模态 ('rgb' 或 'flow')
        dataset: Name of the dataset / 数据集名称
        feature_layer: Layer to extract features from / 提取特征的层
    
    Returns:
        None
    """
    model.eval()
    start_time = timeit.default_timer()
    
    # Determine video name position in file path based on dataset
    # 根据数据集确定文件路径中视频名称的位置
    if dataset=='shanghaitech':
        video_name_po = -2
    else:
        video_name_po = -3
    
    # Process each batch of video clips
    # 处理每个批次的视频片段
    for img, fileinputs in tqdm(dataloader):
        # Move inputs to the device
        # 将输入移动到设备
        inputs = img.to(device)
        fileinputs = np.asarray(fileinputs).transpose((1, 0))
        
        # Extract features without gradient computation
        # 无梯度计算提取特征
        with torch.no_grad():
            features, _ = model(inputs, feature_layer=feature_layer)
        
        # Reshape and convert features to numpy
        # 重塑特征并转换为numpy
        features = features.view(features.size(0), -1)
        features = features.data.cpu().numpy()
        
        # Save features for each video clip
        # 为每个视频片段保存特征
        for (fileinput, feature) in zip(fileinputs, features):
            if datamodal == 'flow' or datamodal == 'flownet':
                video_name = fileinput[0].split(':')[0].split('/')[video_name_po]
                start_frame = fileinput[0].split(':')[0].split('/')[-1].split('.')[0].split('_')[-1]
                end_frame = fileinput[-1].split(':')[0].split('/')[-1].split('.')[0].split('_')[-1]
                save_path = os.path.join(feature_save_dir, video_name, start_frame + '_' +end_frame + '.npy')
            else:
                video_name = fileinput[0].split('/')[video_name_po]
                start_frame = fileinput[0].split('/')[-1].split('.')[0].split('_')[-1]
                end_frame = fileinput[-1].split('/')[-1].split('.')[0].split('_')[-1]
                save_path = os.path.join(feature_save_dir, video_name, start_frame + '_' +end_frame + '.npy')

            # Create video directory if it doesn't exist
            # 如果视频目录不存在则创建
            if os.path.exists(os.path.join(feature_save_dir, video_name)) == 0:
                os.makedirs(os.path.join(feature_save_dir, video_name))
            np.save(save_path, feature)
    
    stop_time = timeit.default_timer()
    print("Execution time: " + str(stop_time - start_time) + "\n")





if __name__ == "__main__":
    """
    Main entry point for feature extraction / 特征提取的主入口点
    
    This script sets up the environment, parses command line arguments, and initiates
    the feature extraction process for video anomaly detection.
    此脚本设置环境，解析命令行参数，并启动视频异常检测的特征提取过程。
    """
    
    # Set device for computation (GPU if available, else CPU)
    # 设置计算设备（如果可用则使用GPU，否则使用CPU）
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)
    
    # Parse command line arguments
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", help='path of testing model_weight', default='./model/i3d/i3d_model_weight/model_kinetics_rgb.pth', type=str)
    parser.add_argument("--datamodal", help='rgb or flow', default='rgb', type=str)
    parser.add_argument("--dataset", help='Name of dataset', default='shanghaitech', type=str)
    parser.add_argument("--modelName", help='Name of model', default='i3d', type=str)
    parser.add_argument("--fc_layer", help='layer of feature extraction', default='fc6', type=str)
    args = parser.parse_args()
    
    # Extract arguments
    # 提取参数
    snapshot = args.snapshot
    Dataset = args.dataset
    datamodal = args.datamodal
    data_path = '/home/tu-wan/windowswan'  # Hardcoded data path, consider making this configurable
                                           # 硬编码的数据路径，考虑使其可配置
    
    # Define file paths for data processing
    # 定义数据处理的文件路径
    origin_filelist = './dataset/{}/{}/{}_list.txt'.format(Dataset,args.modelName, datamodal)
    origin_labellist = './dataset/{}/{}/label.txt'.format(Dataset,args.modelName)
    trainfile_list = './dataset/{}/{}/{}_list_numJoints.txt'.format(Dataset,args.modelName, datamodal)
    trainlabel_list = './dataset/{}/{}/trainlabel_numJoints.txt'.format(Dataset,args.modelName)

    # Process text files for dataset preparation
    # 处理文本文件以准备数据集
    numJoints = 16
    txttans(origin_filelist=origin_filelist,
            origin_labellist=origin_labellist,
            processed_filelist=trainfile_list ,
            processed_labellist=trainlabel_list,
            numJoints=numJoints,
            model='train',
            framework=' ')
    
    # Create dataset and dataloader
    # 创建数据集和数据加载器
    train_dataset = trainDataset(list_file=trainfile_list,
                                 GT_file=trainlabel_list,
                                 transform=None,
                                 cliplen=numJoints,
                                 datamodal=datamodal,
                                 args=args)

    train_dataloader = DataLoader(dataset=train_dataset,batch_size=24, pin_memory=True,
                              num_workers=5,shuffle=False)

    modelName = args.modelName  # Options: C3D or I3D

    # Start feature extraction process
    # 启动特征提取过程
    feature(data_path=data_path,
            dataset=Dataset,
            snapshot=snapshot,
            modelName = modelName,
            dataloader= train_dataloader,
            datamodal= datamodal,
            fc_layer=args.fc_layer)
