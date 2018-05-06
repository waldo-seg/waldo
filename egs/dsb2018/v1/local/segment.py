import torch
import numpy as np
import argparse
import os
import sys
from tqdm import tqdm
import shutil
import time
import torchvision
import random
import PIL
from torchvision import transforms as tsf
from skimage.io import imread
from models.Unet import UNet
from skimage.transform import resize
from skimage.morphology import label

sys.path.insert(0, 'scripts')
from segmenter.segmenter import ObjectSegmenter


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


    testset = Dataset(args.val_data, s_trans, offset_list,
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
    torchvision.utils.save_image(class_id[0, 0, :, :], 'class0.png')  # backgrnd
    torchvision.utils.save_image(class_id[0, 1, :, :], 'class1.png')  # cells
    #sys.exit('stop')

#    model = UNet(2, 2).cuda()
    model = UNet(2, 2)

    # optionally resume from a checkpoint
    if args.model:
        if os.path.isfile(args.model):
            print("=> loading checkpoint '{}'".format(args.model))
            checkpoint = torch.load(args.model)
            model.load_state_dict(checkpoint['state_dict'])
            #model.cpu()
            print("loaded.")
            #model.
            #torch.save({
            #    'epoch': checkpoint['epoch'] + 1,
            #    'state_dict': model.state_dict(),
            #    'best_prec1': checkpoint['best_prec1'],
            #}, "exp/final.tar")
        else:
            print("=> no checkpoint found at '{}'".format(args.model))

    img = torch.autograd.Variable(img)
    predictions = model(img)
    predictions = predictions.data
    class_pred = predictions[0, :args.num_classes, :, :]  # [batch-idx, class-idx, x, y]
    adj_pred = predictions[0, args.num_classes:, :, :]    # [batch-idx, offset-idx, x, y]
    seg = ObjectSegmenter(class_pred.numpy(), adj_pred.numpy(), args.num_classes, offset_list)
#    seg = ObjectSegmenter(class_id[0, :, :, :].numpy(), sameness[0, :, :, :].numpy(), args.num_classes, offset_list)
    seg.run_segmentation()


#    img = torch.autograd.Variable(img).cuda()
#    img = torch.autograd.Variable(img)
#    predictions = model(img)
#    predictions = predictions.data
#    class_pred = predictions[:, :args.num_classes, :, :]  # [batch-idx, class-idx, x, y]
#    adj_pred = predictions[:, args.num_classes:, :, :]    # [batch-idx, offset-idx, x, y]
    for i in range(len(offset_list)):
        torchvision.utils.save_image(
            adj_pred[i, :, :], 'sameness_pred{}.png'.format(i))
    for i in range(args.num_classes):
        torchvision.utils.save_image(
            class_pred[i, :, :], 'class_pred{}.png'.format(i))



class Dataset():
    def __init__(self, path, transformation, offset_list,
                 num_classes, height, width):
        self.data = torch.load(path)
        self.transformation = transformation
        self.offset_list = offset_list
        self.num_classes = num_classes
        self.height = height
        self.width = width

    def __getitem__(self, index):
        data = self.data[index]
        # input images
        img = data['img'].numpy()
        height, width, channel = img.shape
        img = self.transformation(img)

        # bounding box
        num_offsets = len(self.offset_list)
        mask = data['mask'].numpy()
        # np.set_printoptions(threshold='nan')
        # print mask
        bound = torch.zeros(num_offsets, self.height, self.width)

        for k in range(num_offsets):
            i, j = self.offset_list[k]
            rolled_mask = np.roll(np.roll(mask, i, axis=1), j, axis=0)
            bound_unscaled = (torch.FloatTensor(
                (rolled_mask == mask).astype('float'))).unsqueeze(0)
            bound[k:k + 1] = self.transformation(bound_unscaled)

        # class label
        class_label = torch.zeros((self.num_classes, self.height, self.width))
        for c in range(self.num_classes):
            if c == 0:
                class_label_unscaled = (torch.FloatTensor(
                    (mask == 0).astype('float'))).unsqueeze(0)
            else:  # TODO, the current is for 2 classes only
                class_label_unscaled = (torch.FloatTensor(
                    (mask > 0).astype('float'))).unsqueeze(0)
            class_label[c:c +
                        1] = self.transformation(class_label_unscaled)

        return img, class_label, bound

    def __len__(self):
        return len(self.data)



if __name__ == '__main__':
    main()
