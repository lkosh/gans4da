from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import os
import models.dcgan_v2_wcgan as dcganv2
import models.dcgan as dcgan
from torch import autograd

parser = argparse.ArgumentParser()
#parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw ')
#parser.add_argument('--dataroot', required=True, help='path to dataset')
#parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nc', type=int, default=3, help='input image channels')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)

parser.add_argument('--cuda'  , action='store_true',default =True,  help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')

opt = parser.parse_args()
opt.netG = './samples/netG_epoch_499.pth'

PIC4CLASS = 1000
netG = dcganv2._netG(64, opt.nz, opt.nc, opt.ngf, opt.ngpu, 0)
netG.load_state_dict(torch.load(opt.netG))
#netG = torch.load('./samples/netG_epoch_499.pth')
nz = 100
batchsize = 1
noise = torch.FloatTensor(batchsize, nz, 1, 1)
label_array = torch.LongTensor(batchsize, 1)
if opt.cuda:
    netG = netG.cuda()
    noise = noise.cuda()
    label_array = label_array.cuda()
    
if not os.path.exists('./generated_dataset'):
    os.mkdir('./generated_dataset')
for label in range(0, 43):
    path = './generated_dataset/class_'+ str(label)
    if not os.path.exists(path):
        os.mkdir(path)
    for i in range(PIC4CLASS):
    
        noise.resize_(batchsize, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise, volatile = True) # totally freeze netG
        label_array.resize_(batchsize).fill_(label ) # use smooth label for discriminator
        label_array.resize_(label_array.shape[0], 1)
        
        tmp = torch.LongTensor(label_array.shape[0], 43).cuda()
        #print (label.shape, tmp.shape)
        
        tmp.zero_()
        #print (type(label_array), type(tmp))
        
        tmp.scatter_(1, label_array, 1)
        #print (tmp.shape)
        labelv = Variable(tmp.float())
        

        fake = netG(noisev, labelv)
        fake.data = fake.data.mul(0.5).add(0.5)

        vutils.save_image(fake.data, '{0}/image_{1}.png'.format(path, str(i)))
        



                