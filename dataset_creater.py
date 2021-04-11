import numpy as np
import os
import glob
from multiprocessing import Pool
from tqdm import tqdm
import argparse

def combine_feature(video):
    combine_fea = []
    rgb_fea = []
    flow_fea = []
    rgb_features = glob.glob(os.path.join('/home/tu-wan/windowswan/dataset/{}//features//{}'.format(dataset, pretrain_model),'rgb',video, '*.npy'))
    rgb_features.sort(key=lambda x: int(x[-9:-4]))
    flow_features = glob.glob(os.path.join('/home/tu-wan/windowswan/dataset/{}//features//{}'.format(dataset, pretrain_model), 'flow', video, '*.npy'))
    flow_features.sort(key=lambda x: int(x[-9:-4]))
    for i in range(len(rgb_features)):
        rgb_fea_np = np.load(rgb_features[i])
        flow_fea_np = np.load(flow_features[i])
        rgb_fea.append(rgb_fea_np)
        flow_fea.append(flow_fea_np)
        feature = np.hstack((rgb_fea_np,flow_fea_np))
        combine_fea.append(feature)
    combine_fea = np.asarray(combine_fea)
    rgb_fea = np.asarray(rgb_fea)
    flow_fea = np.asarray(flow_fea)
    save_path = os.path.join('/home/tu-wan/windowswan/dataset/{}//features_video//{}'.format(dataset, pretrain_model), 'combine_flownet',video)
    if os.path.exists(save_path) == 0:
        os.makedirs(save_path)
    if os.path.exists(os.path.join('/home/tu-wan/windowswan/dataset/{}//features_video//{}'.format(dataset, pretrain_model), 'rgb', video)) == 0:
        os.makedirs(os.path.join('/home/tu-wan/windowswan/dataset/{}//features_video//{}'.format(dataset, pretrain_model), 'rgb', video))
    if os.path.exists(os.path.join('/home/tu-wan/windowswan/dataset/{}//features_video//{}'.format(dataset, pretrain_model), 'flownet', video)) == 0:
        os.makedirs(os.path.join('/home/tu-wan/windowswan/dataset/{}//features_video//{}'.format(dataset, pretrain_model), 'flownet', video))
    np.save(file=os.path.join(save_path, 'feature.npy'),arr=combine_fea)
    np.save(file=os.path.join('/home/tu-wan/windowswan/dataset/{}//features_video//{}'.format(dataset, pretrain_model), 'rgb', video, 'feature.npy'), arr=rgb_fea)
    np.save(file=os.path.join('/home/tu-wan/windowswan/dataset/{}//features_video//{}'.format(dataset, pretrain_model), 'flownet', video,  'feature.npy'), arr=flow_fea)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help='Name of dataset', default='shanghaitech', type=str)
    args = parser.parse_args()
    pretrain_model = 'i3d'
    dataset = args.dataset
    feature_dir = '/home/tu-wan/windowswan/dataset/{}/features/{}'.format(dataset, pretrain_model)
    # # #
    '''combine the rgb and flow feature on every video'''
    rgb_feature_dir = os.path.join(feature_dir, 'rgb')
    flow_feature_dir = os.path.join(feature_dir, 'flow')
    videos = os.listdir(rgb_feature_dir)
    with Pool(processes=6) as p:
        max_ = len(videos)
        with tqdm(total=max_) as pbar:
            for i, _ in tqdm(enumerate(p.imap_unordered(combine_feature, videos))):
                pbar.update()





