import torch
import time
from utils import AverageMeter, ProgressMeter

def training_one_epoch_MAE(train_loader, model, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.3f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    torch.autograd.set_detect_anomaly(True)
    end = time.time()
    for i, (Signals, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if model.channel_num == 1:
            Signals = Signals[:, 1:2, :]

        if args.gpu is not None:
            Signals = Signals.cuda(args.gpu, non_blocking=True)

        loss = model(Signals, Signals, mask_ratio=0.75)
        losses.update(loss.item(), Signals.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)