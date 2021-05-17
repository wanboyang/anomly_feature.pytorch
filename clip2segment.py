import torch
import torch.nn.functional as f
import os
import glob
import numpy as np
from tqdm import tqdm
clip_features = glob.glob('/home/tu-wan/windowswan/dataset/UCSDPed2/features_video/c3d/fc6/rgb/*/*')

for clip_feature in tqdm(clip_features):
    tmp_f = np.load(clip_feature)
    tmp_f = torch.from_numpy(tmp_f).unsqueeze(0)
    feature_d = tmp_f.shape[-1]
    tar_f = f.adaptive_avg_pool2d(tmp_f, (32,feature_d))
    tar_f = tar_f.squeeze().numpy()
    if os.path.exists(os.path.split(clip_feature.replace('c3d', 'c3d_segments'))[0]) == 0:
        os.makedirs(os.path.split(clip_feature.replace('c3d', 'c3d_segments'))[0])
    np.save(file=clip_feature.replace('c3d', 'c3d_segments'),arr=tar_f)




