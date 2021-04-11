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
    layers = ['pool5']
    for layer in layers:
        rgb_fea = []
        rgb_features = glob.glob(os.path.join('/home/tu-wan/windowswan/dataset/{}//features//{}'.format(dataset, pretrain_model), layer, 'rgb',video, '*.npy'))
        rgb_features.sort(key=lambda x: int(x.split('_')[-1].split('.')[0].zfill(8)))
        for i in range(len(rgb_features)):
            rgb_fea_np = np.load(rgb_features[i])
            rgb_fea.append(rgb_fea_np)
        rgb_fea = np.asarray(rgb_fea)
        if os.path.exists(os.path.join('/home/tu-wan/windowswan/dataset/{}//features_video//{}'.format(dataset, pretrain_model), layer, 'rgb', video)) == 0:
            os.makedirs(os.path.join('/home/tu-wan/windowswan/dataset/{}//features_video//{}'.format(dataset, pretrain_model), layer, 'rgb', video))
        np.save(file=os.path.join('/home/tu-wan/windowswan/dataset/{}//features_video//{}'.format(dataset, pretrain_model), layer, 'rgb', video, 'feature.npy'), arr=rgb_fea)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help='Name of dataset', default='shanghaitech', type=str)
    args = parser.parse_args()
    pretrain_model = 'c3d'
    dataset = args.dataset
    feature_dir = '/home/tu-wan/windowswan/dataset/{}/features/{}'.format(dataset, pretrain_model)
    # layers = os.listdir(feature_dir)
    layer = 'pool5'
    '''combine the rgb and flow feature on every video'''
    rgb_feature_dir = os.path.join(feature_dir, layer, 'rgb')
    # flow_feature_dir = os.path.join(feature_dir, layer, 'flow')
    videos = os.listdir(rgb_feature_dir)
    with Pool(processes=6) as p:
        max_ = len(videos)
        with tqdm(total=max_) as pbar:
            for i, _ in tqdm(enumerate(p.imap_unordered(combine_feature, videos))):
                pbar.update()








