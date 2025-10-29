import argparse
import random
import shutil
import warnings
import os

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, ConcatDataset
import load_data
from training import training_one_epoch_MAE
from models.MAE import MAE_model
from models.resunet import ResUNet_model

parser = argparse.ArgumentParser(description='Multitask PPG MAE')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')

## model parameters
parser.add_argument('-a', '--arch', metavar='ARCH', default='resunet',
                    help='model architecture: ')
parser.add_argument('--resume', default="",
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

## dataset parameters
parser.add_argument('--data-path', default="E:\PluseDB\Subset_Files\CalBased_Test_Subset_no_filter.mat",
                    type=str, help='path to dataset')
parser.add_argument('--data-path_2', default="E:\PluseDB\Supplementary_Subset_Files\VitalDB_CalBased_Test_Subset_no_filter.mat",
                    type=str, help='path to dataset')
parser.add_argument('--data-name', default="PulseDB_VitalDB",
                    type=str, help='data name')
parser.add_argument('--stage', default="pretrain",
                    type=str, help='model training stage')
parser.add_argument('--batch-size', default=256, type=int,
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-j', '--workers', default=1, type=int,
                    help='number of data loading workers')

## training parameters
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--epochs', default=100, type=int, help='max number of total training epochs')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--channel-num', default=1, type=int,
                    help='input channel number')
parser.add_argument('--save-step', default=10, type=int,
                    help='saving model step')
parser.add_argument('--save-path', default='pretrain_model', type=str,
                    help='model saving path')

## optimizer parameters
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=0.05, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

def main():
    args = parser.parse_args()

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


    # load model
    if args.arch == 'MAE':
        model = MAE_model(channel_num=args.channel_num)
    elif args.arch == 'resunet':
        model = ResUNet_model(channel_num=args.channel_num)
    else:
        print(f'do not find the model {args.arch}')

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    optim_params = model.parameters()
    optimizer = torch.optim.AdamW(optim_params, args.lr,
                                  weight_decay=args.weight_decay)

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
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    ## import training data and set dataset, dataloader
    train_dataset = load_data.PPG_dataset(args.data_path, data_name=args.data_name, stage=args.stage)
    train_dataset_2 = load_data.PPG_dataset(args.data_path_2, data_name=args.data_name, stage=args.stage)
    train_dataset = ConcatDataset([train_dataset, train_dataset_2])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    ## train and validate models
    for i in range(args.start_epoch, args.epochs):
        ## train models
        training_one_epoch_MAE(train_loader, model, optimizer, i, args)

        ## saving model
        if (i + 1) % args.save_step == 0:
            save_checkpoint({
                'epoch': i + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=False,
            fold_path=args.save_path, file_name='checkpoint_{:04d}.pth.tar'.format(i))
            # MAE
            PPG, _ = train_dataset[0]
            PPG = PPG.cuda(args.gpu, non_blocking=True).unsqueeze(dim=0)
            model.get_reconstruct_signal(PPG, PPG, i, save_file=args.fig_path)


def save_checkpoint(state, is_best, fold_path='', file_name='checkpoint.pth.tar'):
    path = os.path.join(fold_path, file_name)
    if not os.path.isdir(fold_path):
        os.makedirs(fold_path)
        print(f"folder {fold_path} created")
    path_best = os.path.join(fold_path, 'model_best.pth.tar')
    torch.save(state, path)
    if is_best:
        shutil.copyfile(path, path_best)

if __name__ == '__main__':
    main()