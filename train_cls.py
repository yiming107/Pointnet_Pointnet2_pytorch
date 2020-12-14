"""
Author: Benny
Date: Nov 2019

Modified by: Yiming
Main changes: compatible to train other dataset
"""
from data_utils.ModelNetDataLoader import ModelNetDataLoader
from data_utils.ReplicaDataLoader import ReplicaDataLoader
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
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training [default: 24]')
    parser.add_argument('--dataset_name', type=str, default="replica", help='dataset name to use [default: modelnet40]') #modelnet40_normal_resampled
    parser.add_argument('--model', default='pointnet2_cls_msg', help='model name [default: pointnet_cls]')
    parser.add_argument('--epoch',  default=1000, type=int, help='number of epoch in training [default: 200]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
    parser.add_argument('--log_dir', type=str, default='pointnet2_cls_msg', help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    return parser.parse_args()


def test(model, loader, device, num_class):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.to(device), target.to(device)
        classifier = model.eval()
        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
            class_acc[cat,1]+=1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
    class_acc[:,2] = class_acc[:,0]/ class_acc[:,1]
    class_acc = np.mean(class_acc[:,2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    dataset_name = args.dataset_name
    experiment_dir = experiment_dir.joinpath('classification_{}'.format(dataset_name))
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''TENSORBOARD LOG'''
    writer = SummaryWriter()

    '''DATA LOADING'''
    log_string('Load dataset ...')

    DATA_PATH = os.path.join(ROOT_DIR, 'data', dataset_name)

    print("loading dataset from {}".format(dataset_name))
    if 'modelnet' in dataset_name:
        TRAIN_DATASET = ModelNetDataLoader(DATA_PATH, split='train',
                                                         normal_channel=args.normal)
        TEST_DATASET = ModelNetDataLoader(DATA_PATH, split='test',
                                                    normal_channel=args.normal)
        num_class = 40
    else:
        print(DATA_PATH)
        TRAIN_DATASET = ReplicaDataLoader(DATA_PATH, split='train', uniform=True, normal_channel=False, rot_transform=True)
        TEST_DATASET = ReplicaDataLoader(DATA_PATH, split='test', uniform=True, normal_channel=False, rot_transform=False)
        num_class = 31

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=6)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=6)

    '''MODEL LOADING'''
    print("Number of classes are {:d}".format(num_class))
    MODEL = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('./models/pointnet_util.py', str(experiment_dir))
    
    print("Obtain GPU device ")
    train_GPU = True
    device = torch.device("cuda" if (torch.cuda.is_available() and train_GPU) else "cpu")
    print(device)
    print("Load the network to the device ")
    classifier = MODEL.get_model(num_class, normal_channel=args.normal).to(device)
    print("Load the loss to the device ")
    criterion = MODEL.get_loss().to(device)

    if os.path.exists((str(experiment_dir) + '/checkpoints/best_model.pth')):
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])

         # strict set to false to allow using the model trained with modelnet
    else:
        start_epoch = 0
        if dataset_name == 'replica':
            log_string('Use pretrain model of Model net')
            # double check again if there is pretrained modelnet model
            checkpoint = torch.load(str(experiment_dir).replace("replica", 'modelnet40_normal_resampled')+'/checkpoints/best_model.pth')
            classifier = MODEL.get_model(40, normal_channel=args.normal).to(device)
            classifier.load_state_dict(checkpoint['model_state_dict'])
            classifier.fc3 = nn.Linear(256, num_class).to(device)
            print(classifier)
        else:
            log_string('No existing model, starting training from scratch...')

    if args.optimizer == 'Adam':
        print("Using Adam opimizer ")
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    mean_correct = []

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        loss_array = np.zeros((len(trainDataLoader),1))
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        classifier.train() # setting the model to train mode
        print("Clear GPU cache ...")
        torch.cuda.empty_cache()
        scheduler.step()
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            points, target = data
            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            target = target[:, 0]

            points = points.transpose(2, 1)
            points, target = points.to(device), target.to(device)

            optimizer.zero_grad()

            pred, trans_feat = classifier(points) ### This is the part of the runtime error:

            loss = criterion(pred, target.long(), trans_feat)
            loss_array[batch_id] = loss.cpu().detach().numpy()

            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1

            train_instance_acc = np.mean(mean_correct)
            log_string('Train Instance Accuracy: %f' % train_instance_acc)
        avg_loss = np.mean(loss_array[:])
        writer.add_scalar("Loss/train", avg_loss, epoch)

        ## This is for validation
        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), testDataLoader, device, num_class)

            writer.add_scalar("ClassAccuracy/test", class_acc, epoch)
            writer.add_scalar("InstanceAccuracy/test", instance_acc, epoch)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f'% (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'.format(epoch)
                log_string('Saving at %s'% savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')
    writer.flush()
    writer.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)
