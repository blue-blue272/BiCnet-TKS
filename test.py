from __future__ import print_function, absolute_import
import os
import gc
import sys
import time
import math
import h5py
import scipy
import datetime
import argparse
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import lr_scheduler

import data_manager
from video_loader import VideoDataset
import transforms.spatial_transforms as ST
import transforms.temporal_transforms as TT
import models
from losses import CrossEntropyLabelSmooth, TripletLoss
from utils import AverageMeter, Logger, save_checkpoint
from eval_metrics import evaluate
from samplers import RandomIdentitySampler
from evaluation1 import evaluation

parser = argparse.ArgumentParser(description='Train video model with cross entropy loss')
# Datasets
parser.add_argument('-d', '--dataset', type=str, default='lsvid',
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=256,
                    help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=128,
                    help="width of an image (default: 128)")
# Augment
parser.add_argument('--test_frames', default=4, type=int, help='frames/clip for test')
parser.add_argument('--little-frames', type=int, default=3)
#parser.add_argument('--skip_frames', default=4, type=int, help='frames/clip for skip in test')
# Optimization options
parser.add_argument('--distance', type=str, default='consine', help="euclidean or consine")
parser.add_argument('--num_instances', type=int, default=0, help="number of instances per identity")
# Architecture
parser.add_argument('-a', '--arch', type=str, default='SAresnet50')
parser.add_argument('--pool-mode', type=str, default='max')
parser.add_argument('--save-dir', type=str, default='./result/lsvid/baseline/seq4-stride8-bs32-lr35-m0.3-erase0.5-sa-layer2')
parser.add_argument('--resume', type=str, default='./result/lsvid/baseline/seq4-stride8-bs32-lr35-m0.3-erase0.5-sa-layer2/checkpoint_ep140.pth.tar', metavar='PATH')
# Miscs
parser.add_argument('--last-s1', default=True)
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--use_cpu', action='store_true', help="use cpu")
parser.add_argument('--gpu_devices', default='1', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        #cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing dataset {}".format(args.dataset))
    dataset = data_manager.init_dataset(name=args.dataset)

    # Data augmentation
    spatial_transform_test = ST.Compose([
                ST.Scale((args.height, args.width), interpolation=3),
                ST.ToTensor(),
                ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    pin_memory = True if use_gpu else False

    print("Initializing model: {}".format(args.arch))
    model = models.init_model(name=args.arch, pool_mode=args.pool_mode, last_s1=args.last_s1, 
            little_frames=args.little_frames, num_classes=dataset.num_train_pids, loss={'xent', 'htri'})
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

    if args.resume:
        print("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']

    if use_gpu:
        model = nn.DataParallel(model).cuda()
        # model = model.cuda()

    print("Evaluate only")
    test_all_frames(model, dataset, spatial_transform_test, use_gpu)


def test_all_frames(model, dataset, spatial_transform_test, use_gpu):
    print("==> Start testing the model")
    temporal_transform_test = None

    pin_memory = True if use_gpu else False

    queryloader = DataLoader(
        VideoDataset(dataset.query, spatial_transform=spatial_transform_test, temporal_transform=temporal_transform_test),
        batch_size=1, shuffle=False, num_workers=0,
        pin_memory=pin_memory, drop_last=False
    )

    galleryloader = DataLoader(
        VideoDataset(dataset.gallery, spatial_transform=spatial_transform_test, temporal_transform=temporal_transform_test),
        batch_size=1, shuffle=False, num_workers=0,
        pin_memory=pin_memory, drop_last=False
    )

    if not args.resume:
        print("Loading checkpoint from best_model")
        weights = os.path.join(args.save_dir, 'best_model.pth.tar')
        checkpoint = torch.load(weights)
        model.module.load_state_dict(checkpoint['state_dict'])

    model.eval()
    with torch.no_grad():
        evaluation(model, args, queryloader, galleryloader, use_gpu)



if __name__ == '__main__':
    main()
