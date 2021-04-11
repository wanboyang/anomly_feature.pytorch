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
        Args:
            num_classes (int): Number of classes in the data
            num_epochs (int, optional): Number of epochs to train for.
    """
    ######################build model#####################################
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
    ######################load weigths#####################################
    if snapshot:
        model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(snapshot).items()})
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

    if os.path.exists(feature_save_dir) == 0:
        os.makedirs(feature_save_dir)
######################log#####################################
    if os.path.exists(os.path.join('./model_feature/', dataset, modelName)) == 0:
        os.makedirs(os.path.join('./model_feature/', dataset, modelName))
    with open(file=os.path.join('./model_feature/', dataset, modelName,'feature.txt'), mode='a+') as f:
        f.write("dataset:{} ".format(dataset)+ '\n')
        f.write("snapshot:{} ".format(snapshot) + '\n')
        f.write("savedir:{} ".format(feature_save_dir) + '\n')
        f.write("========================================== " + '\n')

    model_feature(model=model,dataloader=dataloader, feature_save_dir=feature_save_dir,datamodal=datamodal,dataset=dataset, feature_layer=feature_layer)



def model_feature(model, dataloader, feature_save_dir, datamodal, dataset, feature_layer=None):
    model.eval()
    start_time = timeit.default_timer()
    if dataset=='shanghaitech':
        video_name_po = -2
    else:
        video_name_po = -3
    for img, fileinputs in tqdm(dataloader):
        # move inputs and labels to the device the training is taking place on
        inputs = img.to(device)
        # fileinputs = np.asarray(fileinputs)
        fileinputs = np.asarray(fileinputs).transpose((1, 0))
        with torch.no_grad():
            features, _ = model(inputs, feature_layer=feature_layer)
        features = features.view(features.size(0), -1)
        features = features.data.cpu().numpy()
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

            if os.path.exists(os.path.join(feature_save_dir, video_name)) == 0:
                os.makedirs(os.path.join(feature_save_dir, video_name))
            np.save(save_path, feature)
    stop_time = timeit.default_timer()
    print("Execution time: " + str(stop_time - start_time) + "\n")





if __name__ == "__main__":

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", help='path of testing model_weight', default='./model/i3d/i3d_model_weight/model_kinetics_rgb.pth', type=str)
    parser.add_argument("--datamodal", help='rgb or flow', default='rgb', type=str)
    parser.add_argument("--dataset", help='Name of dataset', default='shanghaitech', type=str)
    parser.add_argument("--modelName", help='Name of model', default='i3d', type=str)
    parser.add_argument("--fc_layer", help='layer of feature extraction', default='fc6', type=str)
    args = parser.parse_args()
    snapshot = args.snapshot
    Dataset = args.dataset
    datamodal = args.datamodal
    data_path = '/home/tu-wan/windowswan'
    origin_filelist = './dataset/{}/{}/{}_list.txt'.format(Dataset,args.modelName, datamodal)
    origin_labellist = './dataset/{}/{}/label.txt'.format(Dataset,args.modelName)
    trainfile_list = './dataset/{}/{}/{}_list_numJoints.txt'.format(Dataset,args.modelName, datamodal)
    trainlabel_list = './dataset/{}/{}/trainlabel_numJoints.txt'.format(Dataset,args.modelName)

    numJoints = 16
    txttans(origin_filelist=origin_filelist,
            origin_labellist=origin_labellist,
            processed_filelist=trainfile_list ,
            processed_labellist=trainlabel_list,
            numJoints=numJoints,
            model='train',
            framework=' ')
    train_dataset = trainDataset(list_file=trainfile_list,
                                 GT_file=trainlabel_list,
                                 transform=None,
                                 cliplen=numJoints,
                                 datamodal=datamodal,
                                 args=args)

    train_dataloader = DataLoader(dataset=train_dataset,batch_size=24, pin_memory=True,
                              num_workers=5,shuffle=False)

    modelName = args.modelName  # Options: C3D or I3D

    feature(data_path=data_path,
            dataset=Dataset,
            snapshot=snapshot,
            modelName = modelName,
            dataloader= train_dataloader,
            datamodal= datamodal,
            fc_layer=args.fc_layer)
