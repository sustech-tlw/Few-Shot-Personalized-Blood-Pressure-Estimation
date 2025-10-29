import argparse
import random
import warnings
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import ft_set
import load_data
from meta_training import val_one_epoch_meta
from models.Transformer import Transformer_model
from models.Transformer_LoRA import Transformer_LoRA_model
from models.resunet import ResNet_model

parser = argparse.ArgumentParser(description='Multitask PPG MAE')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')

## model parameters
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet',
                    help='model architecture: ')
parser.add_argument('--resume', default="E:\PPG-MMAE\\ft-model\\10_15-two_dataset-uniform_sampling-meta-learning-multi_channel_two_dataset_mask-cnn-ft_DBP-all-nonLinear_repeat(batch1_woinitial_1e-5)\model_best.pth.tar",
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

## dataset parameters
parser.add_argument('--test-data-path', default="E:\PluseDB\Subset_Files\CalFree_Test_Subset_no_filter.mat",
                    type=str, help='path to the test dataset')
parser.add_argument('--data-name', default="PulseDB_VitalDB",
                    type=str, help='data name')
parser.add_argument('--stage', default="SBP",
                    type=str, help='model training stage')
parser.add_argument('--batch-size', default=1, type=int,
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-j', '--workers', default=1, type=int,
                    help='number of data loading workers')

## training parameters
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--FT-mode', default='all', type=str, help='fine tune mode')
parser.add_argument('--epochs', default=100, type=int, help='max number of total training epochs')
parser.add_argument('--channel-num', default=2, type=int,
                    help='input channel number')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--save-path', default='ft-model', type=str,
                    help='model saving path')


## optimizer parameters
parser.add_argument('--lr', '--learning-rate', default=1, type=float,
                    help='initial (base) learning rate', dest='lr')
parser.add_argument('--meta_inner_lr', default=0.00001, type=float, help='meta learning rate')
parser.add_argument('--meta_inner_step', default=1, type=float, help='meta inner update step')
parser.add_argument(
    '--k_shot', default=5, type=int, help='threshold for early stopping'
)
parser.add_argument(
    '--k_query', default=400, type=int, help='num of query data'
)


best_MAE = np.Inf
best_MSE = np.Inf
def main():
    args = parser.parse_args()
    global best_MAE
    global best_MSE
    ## set initializing seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    ## set deep learning models, optimizer
    if args.arch == 'MAE_multi_channel':
        model = Transformer_model(channel_num=args.channel_num)

    elif args.arch == 'MAE_multi_channel_LoRA':
        model = Transformer_LoRA_model(channel_num=args.channel_num)

    elif args.arch == 'resnet':
        model = ResNet_model(channel_num=args.channel_num)
        model.auxiliary_task = False

    model, params, params_names = ft_set.set_ft_model(model, args)
    print(params_names)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_MAE = checkpoint['best_MAE']
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    ## import training data and set dataset, dataloader
    test_dataset = load_data.PPG_meta_dataset(args.test_data_path, data_name=args.data_name, stage=args.stage, num_person=135)
    # test_dataset = load_private_dataset.private_dataset(args.test_data_path, stage=args.stage, duration_time=570, add_noisy=True)
    # test_dataset = load_BCG_dataset.BCG_dataset(args.test_data_path, data_name='BCG_VMD', stage=args.stage, add_noisy=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

    ## train and validate models
    val_one_epoch_meta(test_loader, model, 0, args, save_name='testing')


if __name__ == '__main__':
    main()