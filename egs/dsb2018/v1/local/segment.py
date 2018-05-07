import torch
import argparse
import os
import sys
import torchvision
import random
from torchvision import transforms as tsf
from models.Unet import UNet
from dataset import Dataset_dsb2018
from waldo.segmenter import ObjectSegmenter


parser = argparse.ArgumentParser(description='Pytorch DSB2018 setup')
parser.add_argument('--model', default='', type=str,
                    help='path to final model (default: none)')
parser.add_argument('--img-height', default=128, type=int,
                    help='Height of resized images')
parser.add_argument('--img-width', default=128, type=int,
                    help='width of resized images')
parser.add_argument('--img-channels', default=3, type=int,
                    help='Number of channels of images')
parser.add_argument('--name', default='Unet-5', type=str,
                    help='name of experiment')
parser.add_argument('--val-data', default='./data/val.pth.tar', type=str,
                    help='Path of processed validation data')
parser.add_argument('--test-data', default='./data/test.pth.tar', type=str,
                    help='Path of processed test data')
parser.add_argument('--num-classes', default=2, type=int,
                    help='Number of classes to classify')


random.seed(0)


def main():
    global args
    args = parser.parse_args()
    args.batch_size = 1
    args.depth = 16

    offset_list = [(1, 1), (0, -2)]

    s_trans = tsf.Compose([
        tsf.ToPILImage(),
        tsf.Resize((args.img_height, args.img_width)),
        tsf.ToTensor(),
    ])

    testset = Dataset_dsb2018(args.val_data, s_trans, offset_list,
                              args.num_classes, args.img_height, args.img_width)
    print('Total samples in the test set: {0}'.format(len(testset)))

    dataloader = torch.utils.data.DataLoader(
        testset, num_workers=1, batch_size=args.batch_size)

    data_iter = iter(dataloader)
#    data_iter.next()
    img, class_id, sameness = data_iter.next()
    torch.set_printoptions(threshold=5000)
    torchvision.utils.save_image(img, 'input.png')
    torchvision.utils.save_image(sameness[0, 0, :, :], 'sameness0.png')
    torchvision.utils.save_image(sameness[0, 1, :, :], 'sameness1.png')
    torchvision.utils.save_image(
        class_id[0, 0, :, :], 'class0.png')  # backgrnd
    torchvision.utils.save_image(class_id[0, 1, :, :], 'class1.png')  # cells

    model = UNet(2, 2)

    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        model.load_state_dict(checkpoint['state_dict'])
        print("loaded.")
    else:
        print("=> no checkpoint found at '{}'".format(args.model))

    model.eval()  # convert the model into evaluation mode

    img = torch.autograd.Variable(img)
    predictions = model(img)
    predictions = predictions.data
    # [batch-idx, class-idx, row, col]
    class_pred = predictions[0, :args.num_classes, :, :]
    # [batch-idx, offset-idx, row, col]
    adj_pred = predictions[0, args.num_classes:, :, :]

    for i in range(len(offset_list)):
        torchvision.utils.save_image(
            adj_pred[i, :, :], 'sameness_pred{}.png'.format(i))
    for i in range(args.num_classes):
        torchvision.utils.save_image(
            class_pred[i, :, :], 'class_pred{}.png'.format(i))

    seg = ObjectSegmenter(class_pred.numpy(),
                          adj_pred.numpy(), args.num_classes, offset_list)
#    seg = ObjectSegmenter(class_id[0, :, :, :].numpy(), sameness[0, :, :, :].numpy(), args.num_classes, offset_list)
    seg.run_segmentation()

    for i in range(len(offset_list)):
        torchvision.utils.save_image(
            adj_pred[i, :, :], 'sameness_pred{}.png'.format(i))
    for i in range(args.num_classes):
        torchvision.utils.save_image(
            class_pred[i, :, :], 'class_pred{}.png'.format(i))


if __name__ == '__main__':
    main()
