import argparse
import random
import shutil
import warnings
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, ConcatDataset

import ft_set
import load_data
from meta_training import training_one_epoch_meta, val_one_epoch_meta
from models.Transformer import Transformer_model
from models.Transformer_LoRA import Transformer_LoRA_model
from models.resunet import ResNet_model

parser = argparse.ArgumentParser(description='Multitask PPG MAE')
parser.add_argument('--seed', default=647, type=int, help='seed for initializing training')

## model parameters
parser.add_argument('-a', '--arch', metavar='ARCH', default='Transformer',
                    help='model architecture: ')
parser.add_argument('--pretrained', default="", type=str, help='path to the pretrained model')
parser.add_argument('--resume', default="",
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

## dataset parameters
parser.add_argument('--data-path', default="E:\PluseDB\Supplementary_Subset_Files\VitalDB_CalFree_Test_Subset_no_filter.mat",
                    type=str, help='path to dataset')
parser.add_argument('--data-path_2', default="E:\PluseDB\Subset_Files\CalFree_Test_Subset_no_filter.mat",
                    type=str, help='path to dataset')
parser.add_argument('--test-data-path', default="E:\PluseDB\Supplementary_Subset_Files\VitalDB_CalFree_Test_Subset_no_filter.mat",
                    type=str, help='path to the test dataset')
parser.add_argument('--test-data-path_2', default="E:\PluseDB\Subset_Files\CalFree_Test_Subset_no_filter.mat",
                    type=str, help='path to the test dataset')
parser.add_argument('--data-name', default="PulseDB_VitalDB",
                    type=str, help='data name')
parser.add_argument('--stage', default="SBP",
                    type=str, help='model training stage')
parser.add_argument('--batch-size', default=5, type=int,
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-j', '--workers', default=0, type=int,
                    help='number of data loading workers')

## training parameters
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--FT-mode', default='all', type=str, help='fine tune mode')
parser.add_argument('--epochs', default=100, type=int, help='max number of total training epochs')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--channel-num', default=2, type=int,
                    help='input channel number')
parser.add_argument('--save-path', default='ft-model', type=str,
                    help='model saving path')
parser.add_argument(
    "--early-stopping", default=10, type=int, help="epoch for early stopping "
)
parser.add_argument(
    '--stop-threshold', default=0.1, type=float, help='threshold for early stopping'
)

## optimizer parameters
parser.add_argument('--lr', '--learning-rate', default=0.00001, type=float,
                    help='initial (base) learning rate', dest='lr')
parser.add_argument('--meta_inner_lr', default=0.00001, type=float, help='meta learning rate')
parser.add_argument('--meta_inner_step', default=1, type=float, help='meta inner update step')
parser.add_argument(
    '--k_shot', default=5, type=int, help='num of support data'
)
parser.add_argument(
    '--k_query', default=400, type=int, help='num of query data'
)
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=0.02, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')


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
        warnings.warn(f'You have chosen to seed {args.seed} training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    ## set deep learning models, optimizer
    if args.arch == 'Transformer':
        model = Transformer_model(channel_num=args.channel_num)

    elif args.arch == 'Transformer_LoRA':
        model = Transformer_LoRA_model(channel_num=args.channel_num)

    elif args.arch == 'resunet':
        model = ResNet_model(channel_num=args.channel_num)
        model.auxiliary_task = False

    model, params, params_names = ft_set.set_ft_model(model, args)
    print(params_names)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    optimizer = torch.optim.AdamW(params, args.lr,
                                  weight_decay=args.weight_decay)

    # load pretrain model
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                if k.startswith('fc'):
                    del state_dict[k]
                else:
                    # if start with encoder., remove encoder.
                    if k.startswith('encoder') and not k.startswith('encoder.fc'):
                        # remove prefix
                        state_dict[k[len("encoder."):]] = state_dict[k]
                        del state_dict[k]
                    elif k.startswith('encoder.fc'):
                        del state_dict[k]

            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            print(msg.missing_keys)

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

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
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    ## import training data and set dataset, dataloader
    train_dataset = load_data.PPG_meta_dataset(args.data_path, data_name=args.data_name, stage=args.stage, num_person=144)
    train_dataset_2 = load_data.PPG_meta_dataset(args.data_path_2, data_name=args.data_name, stage=args.stage, num_person=135)
    train_size = 100
    val_size = len(train_dataset) - train_size
    val_size_2 = len(train_dataset_2) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    del train_dataset
    train_subset_2, val_subset_2 = torch.utils.data.random_split(train_dataset_2, [train_size, val_size_2])
    del train_dataset_2
    train_subset = ConcatDataset([train_subset, train_subset_2])
    val_subset = ConcatDataset([val_subset, val_subset_2])
    test_dataset = load_data.PPG_meta_dataset(args.test_data_path, data_name=args.data_name, stage=args.stage, num_person=144)
    test_dataset_2 = load_data.PPG_meta_dataset(args.test_data_path_2, data_name=args.data_name, stage=args.stage, num_person=135)
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=1, shuffle=True, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader_2 = DataLoader(test_dataset_2, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

    ## train and validate models
    no_imporve = 0
    for i in range(args.start_epoch, args.epochs):
        ## train models
        training_one_epoch_meta(train_loader, model, optimizer, i, args)

        metric = val_one_epoch_meta(val_loader, model, i, args, save_name='validating')
        val_one_epoch_meta(test_loader, model, 0, args, save_name='testing')
        val_one_epoch_meta(test_loader_2, model, 0, args, save_name='testing_2')

        is_best = metric[0] < best_MAE
        # early stop
        if args.early_stopping:
            if metric[4] > best_MSE - args.stop_threshold:
                no_imporve += 1
            else:
                no_imporve = 0
                best_MSE = min(metric[4], best_MSE)
                best_MAE = min(metric[0], best_MAE)
            if no_imporve >= args.early_stopping:
                print('model do not improve {} epoch'.format(args.early_stopping))
                break

        ## saving model
        save_checkpoint({
            'epoch': i + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_MAE': best_MAE,
            'optimizer': optimizer.state_dict(),
        }, is_best, fold_path=args.save_path)

        if i == args.start_epoch and (args.FT_mode == 'last' or args.FT_mode == 'extra_token'):
            params_check(model.state_dict(), args.pretrained)


    # test best model
    best_model_path = os.path.join(args.save_path, 'model_best.pth.tar')
    print("=> loading checkpoint '{}'".format(best_model_path))
    # Map model to be loaded to specified single gpu.
    loc = 'cuda:{}'.format(args.gpu)
    checkpoint = torch.load(best_model_path, map_location=loc)
    args.start_epoch = checkpoint['epoch']
    best_MAE = checkpoint['best_MAE']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(best_model_path, checkpoint['epoch']))

    val_one_epoch_meta(test_loader, model, 0, args, save_name='testing')
    val_one_epoch_meta(test_loader_2, model, 0, args, save_name='testing_2')


def save_checkpoint(state, is_best, fold_path='', file_name='checkpoint.pth.tar'):
    path = os.path.join(fold_path, file_name)
    path_best = os.path.join(fold_path, 'model_best.pth.tar')
    torch.save(state, path)
    if is_best:
        shutil.copyfile(path, path_best)


def params_check(state_dict, pretrained_path):
    print("=> loading '{}' for sanity check".format(pretrained_path))
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    pre_state_dict = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        if k.startswith('expert_layers') or k.startswith('gating_network'):
            continue
        elif k in list(pre_state_dict.keys()):
            assert (state_dict[k].cpu() == pre_state_dict[k]).all(), "{} is changed in linear classifier training.".format(k)

    print("=> sanity check passed.")


if __name__ == '__main__':
    main()