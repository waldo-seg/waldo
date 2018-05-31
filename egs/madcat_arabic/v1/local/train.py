#!/usr/bin/env python3

# Copyright 2018 Johns Hopkins University (author: Yiwen Shao)
# Apache 2.0

""" This script trains the encoding network that the input images are of size
    c * w * h and the output feature maps are of size (num_class + num_offset) * w * h
"""

import sys
import torch
import math
import argparse
import os
import shutil
import time
import torchvision
import random
from torchvision import transforms as tsf
from models.Unet import UNet
from waldo.data_io import WaldoDataset
from waldo.core_config import CoreConfig
from unet_config import UnetConfig


parser = argparse.ArgumentParser(description='Pytorch MADCAT  Arabic')
parser.add_argument('dir', type=str,
                    help='directory of output models and logs')
parser.add_argument('--epochs', default=10, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    help='mini-batch size (default: 16)')
parser.add_argument('--train-image-size', default=128, type=int,
                    help='The size of the parts of training images that we'
                    'train on (in order to form a fixed minibatch size).'
                    'These are derived from the input images'
                    ' by padding and then random cropping.')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True,
                    type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--train-dir', default='data', type=str,
                    help='Directory of processed training and validation data')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_false')
parser.add_argument('--core-config', default='', type=str,
                    help='path of core configuration file')
parser.add_argument('--unet-config', default='', type=str,
                    help='path of network configuration file')


best_loss = 1
random.seed(0)


def main():
    global args, best_loss
    args = parser.parse_args()

    if args.tensorboard:
        from tensorboard_logger import configure
        print("Using tensorboard")
        configure("%s" % (args.dir))

    # loading core configuration
    c_config = CoreConfig()
    if args.core_config == '':
        print('No core config file given, using default core configuration')
    if not os.path.exists(args.core_config):
        sys.exit('Cannot find the config file: {}'.format(args.core_config))
    else:
        c_config.read(args.core_config)
        print('Using core configuration from {}'.format(args.core_config))

    # loading Unet configuration
    u_config = UnetConfig()
    if args.unet_config == '':
        print('No unet config file given, using default unet configuration')
    if not os.path.exists(args.unet_config):
        sys.exit('Cannot find the unet configuration file: {}'.format(
            args.unet_config))
    else:
        # need train_image_size for validation
        u_config.read(args.unet_config, args.train_image_size)
        print('Using unet configuration from {}'.format(args.unet_config))

    offset_list = c_config.offsets
    print("offsets are: {}".format(offset_list))

    # model configurations from core config
    num_classes = c_config.num_classes
    num_colors = c_config.num_colors
    num_offsets = len(c_config.offsets)
    # model configurations from unet config
    start_filters = u_config.start_filters
    up_mode = u_config.up_mode
    merge_mode = u_config.merge_mode
    depth = u_config.depth

    train_data = args.train_dir + '/train'
    val_data = args.train_dir + '/dev'

    trainset = WaldoDataset(train_data, c_config, args.train_image_size, crop=False)
    trainloader = torch.utils.data.DataLoader(
        trainset, num_workers=4, batch_size=args.batch_size, shuffle=True)

    valset = WaldoDataset(val_data, c_config, args.train_image_size, crop=False)
    valloader = torch.utils.data.DataLoader(
        valset, num_workers=4, batch_size=args.batch_size)

    NUM_TRAIN = len(trainset)
    NUM_VAL = len(valset)
    NUM_ALL = NUM_TRAIN + NUM_VAL
    print('Total samples: {0} \n'
          'Using {1} samples for training, '
          '{2} samples for validation'.format(NUM_ALL, NUM_TRAIN, NUM_VAL))

    # create model
    model = UNet(num_classes, num_offsets,
                 in_channels=num_colors, depth=depth,
                 start_filts=start_filters,
                 up_mode=up_mode,
                 merge_mode=merge_mode).cuda()

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # define optimizer
    # optimizer = t.optim.Adam(model.parameters(), lr=1e-3)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum, nesterov=args.nesterov,
                                weight_decay=args.weight_decay)

    # Train
    for epoch in range(args.start_epoch, args.epochs):
        Train(trainloader, model, optimizer, epoch)
        val_loss = Validate(valloader, model, epoch)
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_loss,
        }, is_best)
    print('Best validation loss: ', best_loss)

    # visualize some example outputs
    outdir = '{}/imgs'.format(args.dir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    sample(model, valloader, outdir, c_config)


def Train(trainloader, model, optimizer, epoch):
    """Train for one epoch on the training set"""
    losses = AverageMeter()
    batch_time = AverageMeter()

    end = time.time()
    for i, (input, class_label, bound) in enumerate(trainloader):
        adjust_learning_rate(optimizer, epoch + 1)
        input = input.cuda()
        bound = bound.cuda(async=True)
        class_label = class_label.cuda(async=True)

        optimizer.zero_grad()
        output = model(input)
        # class_pred = o[:, :num_classes, :, :]
        # bound_pred = o[:, num_classes:, :, :]
        # TODO. Treat class label and bound label equally by now
        target = torch.cat((class_label, bound), 1)
        loss_fn = torch.nn.BCELoss()
        loss = loss_fn(output, target)

        losses.update(loss.item(), args.batch_size)

        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                      epoch, i, len(trainloader), batch_time=batch_time,
                      loss=losses))

    # log to TensorBoard
    if args.tensorboard:
        from tensorboard_logger import log_value
        log_value('train_loss', losses.avg, epoch)


def Validate(validateloader, model, epoch):
    """Perform validation on the validation set"""
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (input, class_label, bound) in enumerate(validateloader):

        with torch.no_grad():
            input = input.cuda()
            bound = bound.cuda(async=True)
            class_label = class_label.cuda(async=True)
            output = model(input)

            # TODO. Treat class label and bound label equally by now
            target = torch.cat((class_label, bound), 1)
            loss_fn = torch.nn.BCELoss()
            loss = loss_fn(output, target)

            losses.update(loss.item(), args.batch_size)

            if i % args.print_freq == 0:
                print('Val: [{0}][{1}/{2}]\t'
                      'Val Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                          epoch, i, len(validateloader), loss=losses))

    # log to TensorBoard
    if args.tensorboard:
        from tensorboard_logger import log_value
        log_value('val_loss', losses.avg, epoch)

    return losses.avg


def sample(model, dataloader, outdir, core_config):
    """Visualize some predicted masks on training data to get a better intuition
       about the performance.
    """
    data_iter = iter(dataloader)
    img, classification, bound = data_iter.next()
    torchvision.utils.save_image(img, '{0}/raw.png'.format(outdir))
    for i in range(len(core_config.offsets)):
        torchvision.utils.save_image(
            bound[:, i:i + 1, :, :], '{0}/bound_{1}.png'.format(outdir, i))
    for i in range(core_config.num_classes):
        torchvision.utils.save_image(
            classification[:, i:i + 1, :, :], '{0}/class_{1}.png'.format(outdir, i))
    if next(model.parameters()).is_cuda:
        img = img.cuda()
    with torch.no_grad():
        predictions = model(img)
    predictions = predictions.detach()
    class_pred = predictions[:, :core_config.num_classes, :, :]
    bound_pred = predictions[:, core_config.num_classes:, :, :]
    for i in range(len(core_config.offsets)):
        torchvision.utils.save_image(
            bound_pred[:, i:i + 1, :, :], '{0}/bound_pred{1}.png'.format(outdir, i))
    for i in range(core_config.num_classes):
        torchvision.utils.save_image(
            class_pred[:, i:i + 1, :, :], '{0}/class_pred{1}.png'.format(outdir, i))

    return img, class_pred, bound_pred


def soft_dice_loss(inputs, targets):
    num = targets.size(0)
    m1 = inputs.view(num, -1)
    m2 = targets.view(num, -1)
    intersection = (m1 * m2)
    score = 2. * (intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1)
    score = 1 - score.sum() / num
    return score


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR divided by 5 at 60th and 100th epochs"""
    lr = args.lr * ((0.2 ** int(epoch >= 60)) *
                    (0.2 ** int(epoch >= 100)))
    # log to TensorBoard
    if args.tensorboard:
        from tensorboard_logger import log_value
        log_value('learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "%s/" % (args.dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '%s/' %
                        (args.dir) + 'model_best.pth.tar')


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


if __name__ == '__main__':
    main()

