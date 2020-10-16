''' This script is to prepare feature vectors that will be used to graph neural network
We use resnet101 for feature extractor
Author: Yiming Wang (yiming.wang@iit.it)
Last updated: July 2020

This script should be run after the H5 files are prepared
'''

import torch.nn as nn
from data_utils.ReplicaDataLoader import ReplicaDataLoader
from torch.utils.data import DataLoader
import numpy as np
import os
import torch
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import provider
import importlib
import shutil
import torch.nn as nn
import h5py

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
proj_path = BASE_DIR
sys.path.append(os.path.join(proj_path, 'models'))

dataset_name = 'replica'

src_model_folder = os.path.join(proj_path,'log','classification_{}'.format(dataset_name), 'pointnet2_cls_msg')
src_network_path = os.path.join(src_model_folder, 'point')
src_model_path = os.path.join(src_model_folder,'checkpoints','best_model.pth')

num_class = 31
MODEL = importlib.import_module('pointnet2_cls_msg')

print("Obtain GPU device ")
train_GPU = True
device = torch.device("cuda" if (torch.cuda.is_available() and train_GPU) else "cpu")
print("Load the network to the device ")
classifier = MODEL.get_model(num_class, normal_channel=False).to(device)
print(classifier)
print("Load the loss to the device ")
criterion = MODEL.get_loss().to(device)

checkpoint = torch.load(src_model_path)
classifier.load_state_dict(checkpoint['model_state_dict'])

classifier.eval()

DATA_PATH = os.path.join(proj_path, 'data', 'replica')
'''Load dataset '''
TRAIN_DATASET = ReplicaDataLoader(DATA_PATH, split='train', uniform=True, normal_channel=False, rot_transform=False)
TEST_DATASET = ReplicaDataLoader(DATA_PATH, split='test', uniform=True, normal_channel=False, rot_transform=False)
num_train = len(TRAIN_DATASET)
num_test = len(TEST_DATASET)
num_all = num_train + num_test

# prepare the h5 file for save all the features
output_h5_file = os.path.join(DATA_PATH, 'replica_3d_with_feature.h5')

with h5py.File(output_h5_file, 'w') as f:
    feature_dset = None
    with torch.no_grad():
        for dataset_item in [TRAIN_DATASET, TEST_DATASET]:
            data_loader = torch.utils.data.DataLoader(dataset_item,  batch_size=1, shuffle=False, num_workers=6)
            for idx, data in tqdm(enumerate(data_loader), total=len(data_loader)):
                points, target = data
                target = target[:, 0]
                points = points.transpose(2, 1)
                points, target = points.to(device), target.to(device)
                output, feat = classifier(points)
                B, _, _ = points.shape
                feature_vector = feat.view(B, 1024).cpu().numpy()
                target = target.cpu()
                points = points.cpu().numpy()

                torch.cuda.empty_cache()

                if feature_dset == None:
                    points_dset = f.create_dataset('points', (num_all,3, 1024), dtype=np.float64)
                    feature_dset = f.create_dataset('pointnet2_feature', (num_all, 1024), dtype=np.float64)
                    labels_dset = f.create_dataset('labels', (num_all, 1), dtype= np.int8)
                points_dset[idx] = points[0, :, :]
                feature_dset[idx] = feature_vector
                labels_dset[idx] = target[0]


