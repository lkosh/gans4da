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
import models.dcgan_backup as dcgan
#import models.dcgan as dcgan
from torch import autograd

parser = argparse.ArgumentParser()

parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nc', type=int, default=3, help='input image channels')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')

parser.add_argument('--cuda'  , action='store_true',default =True,  help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')

opt = parser.parse_args()
#opt.netG = './samples/netG_epoch_499.pth'
nz = 100
batchsize = 1
PIC4CLASS = 100
netG = dcgan.DCGAN_G(opt.imageSize, nz, opt.nc, opt.ngf, opt.ngpu, opt.n_extra_layers)
#netG.load_state_dict(torch.load(opt.netG))
#netG = torch.load('./samples/netG_epoch_499.pth')

noise = torch.FloatTensor(batchsize, nz, 1, 1)
label_array = torch.LongTensor(batchsize, 1)
if opt.cuda:
    netG = netG.cuda()
    noise = noise.cuda()
    label_array = label_array.cuda()
    
if not os.path.exists('./generated_dataset'):
    os.mkdir('./generated_dataset')
for label in range(0, 43):
    netG.load_state_dict(torch.load('./wgan_models/class_'+str(label) + '/netG_epoch_1999.pth'))

    path = '../data/WGAN_generated/'+ '0'*(5-len(str(label))) + str(label)
    if not os.path.exists(path):
        os.mkdir(path)
    for i in range(PIC4CLASS):
    
        noise.resize_(batchsize, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise, volatile = True) # totally freeze netG  

        fake = netG(noisev)
        fake.data = fake.data.mul(0.5).add(0.5)
        filename = '0'*(5-len(str(label))) + str(label) + '_' + '0'*(5-len(str(i))) + str(i)
        vutils.save_image(fake.data, '{0}/{1}.ppm'.format(path, str(i)))
        



                