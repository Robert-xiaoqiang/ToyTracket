import argparse
import shutil
import os
from os.path import join, isdir, isfile
from os import makedirs

import torch
from torch.utils.data import dataloader
import torch.nn as nn
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import numpy as np
np.random.seed(65535)
import random
random.seed(65535)
import time
import pdb
from tensorboardX import SummaryWriter

from .dataset import VID, MGTVTrainVID
from .net import TopModel
from ..track.UDT_points import DCFNetTracker

def gaussian_shaped_labels(sigma, sz):
    x, y = np.meshgrid(np.arange(1, sz[0]+1) - np.floor(float(sz[0]) / 2), np.arange(1, sz[1]+1) - np.floor(float(sz[1]) / 2))
    d = x ** 2 + y ** 2
    g = np.exp(-0.5 / (sigma ** 2) * d)
    g = np.roll(g, int(-np.floor(float(sz[0]) / 2.) + 1), axis=0)
    g = np.roll(g, int(-np.floor(float(sz[1]) / 2.) + 1), axis=1)
    return g.astype(np.float32)

def output_drop(output, target):
    delta1 = (output - target)**2
    batch_sz = delta1.shape[0]
    delta = delta1.view(batch_sz, -1).sum(dim=1)
    sort_delta, index = torch.sort(delta, descending=True) 
    # unreliable samples (10% of the total) do not produce grad (we simply copy the groundtruth label)
    for i in range(int(round(0.1*batch_sz))):
        output[index[i],...] = target[index[i],...]
    return output

def output_drop_list(output, target):
    ret = [ ]
    for i in range(len(output)):
        ret.append(output_drop(output[i], target[i]))

    return tuple(ret)

class TrackerConfig(object):
    crop_sz = 125
    output_sz = 121

    lambda0 = 1e-4
    padding = 2.0
    output_sigma_factor = 0.1

    output_sigma = crop_sz / (1 + padding) * output_sigma_factor
    y = gaussian_shaped_labels(output_sigma, [output_sz, output_sz])
    yf = torch.rfft(torch.Tensor(y).view(1, 1, output_sz, output_sz).cuda(), signal_ndim=2)
    # cos_window = torch.Tensor(np.outer(np.hanning(crop_sz), np.hanning(crop_sz))).cuda()  # train without cos window

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        save_best_checkpoint(state, os.path.dirname(filename))

def save_best_checkpoint(state, dirname):
    torch.save(state, os.path.join(dirname, 'model_best.pth.tar'))

def list_loss(criterion, output, target, args):
    ret = 0.0
    for i in range(len(output)):
        # print(output[i], target[i])
        ret += criterion(output[i], target[i])
    ret /= args.batch_size

    return ret

def train(train_loader, model, criterion, optimizer, epoch, writer, args, config):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    target = torch.FloatTensor(config.y).cuda().unsqueeze(0).unsqueeze(0).repeat(args.batch_size, 1, 1, 1)  # for training
    # target is the same for 4 points now
    target = [ target ] * 4
    end = time.time()
    num_train_batch_per_epoch = len(train_loader)
    for i, (template, search1, search2) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        template = template.cuda(non_blocking=True)
        search1 = search1.cuda(non_blocking=True)
        search2 = search2.cuda(non_blocking=True)
        output = model(template, search1, search2)

        # target is real, whereas label is complex
        output = output_drop_list(output, target)  # the sample dropout is necessary, otherwise we find the loss tends to become unstable
        
        loss = list_loss(criterion, output, target, args)
        # measure accuracy and record loss
        losses.update(loss.item())
        writer.add_scalar('train/loss_avg', losses.avg, num_train_batch_per_epoch * epoch + i)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {losses.val:.8f} ({losses.avg:.8f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, losses=losses))
    return losses.avg


def validate(tracker):
    dataset = 'mgtv_val'
    gt_root_path = '/home/xqwang/projects/tracking/datasets/mgtv/val'
    data_root_path = '/home/xqwang/projects/tracking/datasets/mgtv/val_preprocessed'
    result_output_path = '/home/xqwang/projects/tracking/UDT/result' 
    return tracker.validate(dataset, data_root_path, gt_root_path, result_output_path)

def get_args():
    parser = argparse.ArgumentParser(description='Training DCFNet in Pytorch 0.4.0')
    parser.add_argument('--input_sz', dest='input_sz', default=125, type=int, help='crop input size')
    parser.add_argument('--padding', dest='padding', default=2.0, type=float, help='crop padding size')
    parser.add_argument('--range', dest='range', default=10, type=int, help='select range')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('-b', '--batch-size', default=48, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--lr-decay', default=0.9, type=float,
                        metavar='LR-DECAY', help='learning rate poly decay rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    # parser.add_argument('--resume', default='/home/xqwang/projects/tracking/UDT/snapshots/crop_125_2.0/checkpoint.pth.tar', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    # parser.add_argument('--save', '-s', default='./work', type=str, help='directory for saving')

    args = parser.parse_args()

    print(args)
    return args
    
def train_main():
    args = get_args()
    gpu_num = torch.cuda.device_count()
    print('GPU NUM: {:2d}'.format(gpu_num))
    
    args.replicata_batch_size = args.batch_size
    args.batch_size *= gpu_num
    args.gpu_num = gpu_num

    config = TrackerConfig()

    model = TopModel(args, config)
    model.cuda()
    model = torch.nn.DataParallel(model, list(range(gpu_num)))

    criterion = nn.MSELoss(reduction = 'mean').cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    lamb = lambda curr: pow((1 - float(curr) / args.epochs), args.lr_decay)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lamb)

    # training data
    crop_base_path = join('/home/xqwang/projects/tracking/datasets/mgtv/train_preprocessed', 'crop_{:d}_{:.1f}'.format(args.input_sz, args.padding))
    json_file_name = join('/home/xqwang/projects/tracking/datasets/mgtv/train_preprocessed', 'dataset.json')
    experiment_key = 'onebyone'
    save_path = join('/home/xqwang/projects/tracking/UDT/snapshots', experiment_key, 'crop_{:d}_{:1.1f}'.format(args.input_sz, args.padding))
    os.makedirs(save_path, exist_ok = True)
    resume_path = join(save_path, 'checkpoint.pth.tar')
    # optionally resume from a checkpoint
    if isfile(resume_path):
        print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(resume_path)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        print("=> loaded checkpoint '{}' (epoch {})"
            .format(resume_path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume_path))


    summary_path = os.path.join('/home/xqwang/projects/tracking/UDT/summary', experiment_key)
    os.makedirs(summary_path, exist_ok = True)
    writer = SummaryWriter(summary_path)

    train_dataset = MGTVTrainVID(file=json_file_name, root=crop_base_path, range=args.range)
    val_dataset = None

    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = None

    best_mse = 2147483647.0
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, epoch, writer, args, config)
        lr_scheduler.step(epoch + 1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict()
        }, False, join(save_path, 'checkpoint.pth.tar'))

        tracker = DCFNetTracker(os.path.join(save_path, 'checkpoint.pth.tar'))
        # evaluate on validation set
        mse = validate(tracker)
        writer.add_scalar('val/mse', mse, epoch)
        # remember best loss and save checkpoint
        is_best = mse < best_mse
        best_mse = min(best_mse, mse)
        if is_best:
            save_best_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict()
            }, save_path)
    writer.close()

if __name__ == '__main__':
    main()
