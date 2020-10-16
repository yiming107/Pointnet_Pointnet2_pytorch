"""
Just used to save a model that is compatible with pytorch < 1.6.0
Date: Nov 2019
"""
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import argparse
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
from torch.utils.tensorboard import SummaryWriter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR

src_model_folder = os.path.join(ROOT_DIR,'log','classification', 'pointnet2_cls_msg')
src_network_path = os.path.join(src_model_folder, 'point')
src_model_path = os.path.join(src_model_folder,'checkpoints','best_model.pth')


# load model

'''MODEL LOADING'''
num_class = 40
MODEL = importlib.import_module('pointnet2_cls_msg')

print("Obtain GPU device ")
train_GPU = True
device = torch.device("cuda" if (torch.cuda.is_available() and train_GPU) else "cpu")
print(device)
print("Load the network to the device ")
classifier = MODEL.get_model(num_class, normal_channel=False).to(device)
print("Load the loss to the device ")
criterion = MODEL.get_loss().to(device)

classifier.eval()

# load optimizer

print("Using Adam opimizer ")
optimizer = torch.optim.Adam(
    classifier.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=1e-4
)

checkpoint = torch.load(src_model_path)
classifier.load_state_dict(checkpoint['model_state_dict'])
saved_epoch = checkpoint['epoch']
print(saved_epoch)
saved_instance_acc = checkpoint['instance_acc']
print(saved_instance_acc)
saved_class_acc = checkpoint['class_acc']
print(saved_class_acc)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# save to compatible version
savepath = os.path.join(src_model_folder, 'checkpoints', 'best_model_compatible.pth')

state = {
    'model_state_dict': classifier.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': saved_epoch,
    'instance_acc': saved_instance_acc,
    'class_acc': saved_class_acc,
    'optimizer_state_dict': optimizer.state_dict(),
}
torch.save(state, savepath, _use_new_zipfile_serialization = False)
