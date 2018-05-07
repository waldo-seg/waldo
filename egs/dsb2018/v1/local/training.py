#!/usr/bin/env python3

# Copyright 2018 Johns Hopkins University (author: Yiwen Shao)
# Apache 2.0

""" This script trains the encoding network that the input images are of size
    c * w * h and the output feature maps are of size (num_class + num_offset) * w * h
"""

import torch
import argparse
import os
import shutil
import time
import torchvision
import random
from torchvision import transforms as tsf
from models.Unet import UNet
from dataset import Dataset_dsb2018

parser = argparse.ArgumentParser(description='Pytorch DSB2018 setup')
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
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True,
                    type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--depth', default=5, type=int,
                    help='Number of conv blocks.')
parser.add_argument('--img-height', default=128, type=int,
                    help='Height of resized images')
parser.add_argument('--img-width', default=128, type=int,
                    help='width of resized images')
parser.add_argument('--img-channels', default=3, type=int,
                    help='Number of channels of images')
parser.add_argument('--name', default='Unet-5', type=str,
                    help='name of experiment')
parser.add_argument('--train-dir', default='data/train_val/split0.9_seed0', type=str,
                    help='Directory of processed training and validation data')
parser.add_argument('--test-dir', default='data/test', type=str,
                    help='Directory of processed test data')
parser.add_argument('--num-classes', default=2, type=int,
                    help='Number of classes to classify')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_false')


best_loss = 1
random.seed(0)


def main():
    global args, best_loss
    args = parser.parse_args()

    if args.tensorboard:
        from tensorboard_logger import configure
        print("Using tensorboard")
        configure("exp/%s" % (args.name))

    s_trans = tsf.Compose([
        tsf.ToPILImage(),
        tsf.Resize((args.img_height, args.img_width)),
        tsf.ToTensor(),
    ])

    offset_list = [(1, 1), (0, -2)]

    train_data = args.train_dir + '/' + 'train.pth.tar'
    val_data = args.train_dir + '/' + 'val.pth.tar'

    trainset = Dataset_dsb2018(train_data, s_trans, offset_list,
                               args.num_classes, args.img_height, args.img_width)
    trainloader = torch.utils.data.DataLoader(
        trainset, num_workers=1, batch_size=args.batch_size, shuffle=True)

    valset = Dataset_dsb2018(val_data, s_trans, offset_list,
                             args.num_classes, args.img_height, args.img_width)
    valloader = torch.utils.data.DataLoader(
        valset, num_workers=1, batch_size=args.batch_size)

    NUM_TRAIN = len(trainset)
    NUM_VAL = len(valset)
    NUM_ALL = NUM_TRAIN + NUM_VAL
    print('Total samples: {0} \n'
          'Using {1} samples for training, '
          '{2} samples for validation'.format(NUM_ALL, NUM_TRAIN, NUM_VAL))

    # create model
    model = UNet(args.num_classes, len(offset_list),
                 in_channels=3, depth=args.depth).cuda()

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
    outdir = 'exp/{}/imgs'.format(args.name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    sample(model, valloader, offset_list, outdir)

    # # Load the best model and evaluate on test set
    # checkpoint = torch.load('exp/%s/' %
    #                         (args.name) + 'model_best.pth.tar')
    # model.load_state_dict(checkpoint['state_dict'])


def Train(trainloader, model, optimizer, epoch):
    """Train for one epoch on the training set"""
    losses = AverageMeter()
    batch_time = AverageMeter()

    end = time.time()
    for i, (input, class_label, bound) in enumerate(trainloader):
        adjust_learning_rate(optimizer, epoch + 1)
        input = torch.autograd.Variable(input.cuda())
        bound = torch.autograd.Variable(bound.cuda(async=True))
        class_label = torch.autograd.Variable(class_label.cuda(async=True))

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

        input = torch.autograd.Variable(input.cuda())
        bound = torch.autograd.Variable(bound.cuda(async=True))
        class_label = torch.autograd.Variable(
            class_label.cuda(async=True))
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


def sample(model, dataloader, offset_list, outdir):
    """Visualize some predicted masks on training data to get a better intuition
       about the performance.
    """
    datailer = iter(dataloader)
    img, classification, bound = datailer.next()
    torchvision.utils.save_image(img, '{0}/raw.png'.format(outdir))
    for i in range(len(offset_list)):
        torchvision.utils.save_image(
            bound[:, i:i + 1, :, :], '{0}/bound_{1}.png'.format(outdir, i))
    for i in range(args.num_classes):
        torchvision.utils.save_image(
            classification[:, i:i + 1, :, :], '{0}/class_{1}.png'.format(outdir, i))
    img = torch.autograd.Variable(img).cuda()
    predictions = model(img)
    predictions = predictions.data
    class_pred = predictions[:, :args.num_classes, :, :]
    bound_pred = predictions[:, args.num_classes:, :, :]
    for i in range(len(offset_list)):
        torchvision.utils.save_image(
            bound_pred[:, i:i + 1, :, :], '{0}/bound_pred{1}.png'.format(outdir, i))
    for i in range(args.num_classes):
        torchvision.utils.save_image(
            class_pred[:, i:i + 1, :, :], '{0}/class_pred{1}.png'.format(outdir, i))


def soft_dice_loss(inputs, targets):
    num = targets.size(0)
    m1 = inputs.view(num, -1)
    m2 = targets.view(num, -1)
    intersection = (m1 * m2)
    score = 2. * (intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1)
    score = 1 - score.sum() / num
    return score


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
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
    directory = "exp/%s/" % (args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'exp/%s/' %
                        (args.name) + 'model_best.pth.tar')


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
