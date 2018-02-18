
# coding: utf-8

# In[1]:
from __future__ import print_function

import glob
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
import numpy as np
import torch.nn.functional as F




# batchsize =64
# imsize=64

parser = argparse.ArgumentParser()
import argparse
parser.add_argument('--gpuid',default=2, help='gpu id')
parser.add_argument('--blend_gan', default=True, help='True, if training should be done on a mixture of original\
                                                        and generated datasets')

parser.add_argument('--gan_dataset',default='./generated_dataset', help='Path to gan-generated dataset, if required')

parser.add_argument('--train_dataset', default='../data/GTSRB/Final_Training/Images', help='Path to a training dataset')
parser.add_argument('--test_dataset', default='../data/GTSRB/Final_Test/Images', help='Path to a test dataset, if required')
parser.add_argument('--batchsize',default=64, help='batchsize, default=64')
parser.add_argument('--imsize',default=64, help='image size, default=64')

opt = parser.parse_args()



fake_dataset = dset.ImageFolder(root='./generated_dataset',
                           transform=transforms.Compose([
                               transforms.Scale(opt.imsize),
                               transforms.CenterCrop(opt.imsize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

train_dataset = dset.ImageFolder(root='../data/GTSRB/Final_Training/Images',
                           transform=transforms.Compose([
                               transforms.Scale(opt.imsize),
                               transforms.CenterCrop(opt.imsize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

test_dataset = dset.ImageFolder(root='../data/GTSRB/Final_Test/Images',
                           transform=transforms.Compose([
                               transforms.Scale(opt.imsize),
                               transforms.CenterCrop(opt.imsize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))



# In[4]:
if opt.blend_gan:
                    
    train_dataset = torch.utils.data.ConcatDataset([train_dataset, fake_dataset])
dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchsize,
                                                  
                                         shuffle=True, num_workers=1)
dataloader_test = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchsize,
                                                  
                                         shuffle=True, num_workers=1)

# In[5]:

nc = 3
ndf = 64
ngpu=1
n_classes = 43
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
torch.cuda.set_device(opt.gpuid)


# In[6]:

class SimpleClassifier(nn.Module):
    def __init__(self, isize, nc, ndf, ngpu, n_classes, n_extra_layers=0):
        super(SimpleClassifier, self).__init__()
        self.ngpu = ngpu
        self.ndf = ndf
        self.n_classes = n_classes
        
        self.main = nn.Sequential(
            # input is (nc) x 96 x 96
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), # 5,3,1 for 96x96
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
        )
        #self.main.add_module('final_layers.conv', nn.Conv2d(ndf * 8, 500, 4, 1, 0, bias=False))
        self.main.add_module('flatten', Flatten())
        #self.main.add_module('final_layers.fc', nn.Linear(ndf * 8, self.n_classes))
        #self.main.add_module('final_layers.softmax', nn.Softmax())

    def forward(self, input):
       
        output = self.main(input)
        n_features = output.size(1)
        fc = nn.Linear(n_features, self.n_classes).cuda(2)
        output =fc(output) #F.log_softmax(fc(output))
            
        return output.view(-1, self.n_classes)




# In[136]:
#
#get_ipython().magic(u'pinfo nn.Conv2d')


# In[6]:

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(2704, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 43)
        self.relu = nn.LeakyReLU(0.2)


    def forward(self, x):
        batchsize = x.size(0)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(batchsize, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net().cuda(opt.gpuid)


# In[131]:




# In[7]:

nepoch = 100
criterion = nn.CrossEntropyLoss()
from keras.utils.np_utils import to_categorical
#opt = optim.Adam(simple_model.parameters(),  lr=0.01)
opt = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
for epoch in range(nepoch):
    train_iter = iter(dataloader_train)
    test_iter = iter(dataloader_test)
    i = 0
    losses = []
    running_loss = 0.0

    while i < len(dataloader_train):
        
        data = train_iter.next()
        net.zero_grad()
        X, y = data
        X = Variable(X).cuda()
        
        y = Variable(y).cuda()
        #print (y_pred.data[0])
        y_pred = net(X)
        loss = criterion( y_pred, y)
        loss.backward()
        opt.step()
        #losses.append(loss.cpu().data.numpy())
        running_loss += loss.cpu().data.numpy()
        if i % 100 == 99:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.5f' %
                  (epoch + 1, i + 1, loss.cpu().data.numpy() ))
            running_loss = 0.0
        i += 1
    i = 0
    test_loss = 0
    acc = 0
    n = 0
    while i<len(dataloader_test):
        data = test_iter.next()
        X, y = data
        X = Variable(X).cuda()
        
        y = Variable(y).cuda()
        y_pred = net(X)
        #loss = criterion( y_pred, y)
        #test_loss += loss.cpu().data.numpy()
        i += 1
        _, classes = y_pred.max(1)
        tmp = y == classes
        n += y.size(0)
        acc += np.sum(tmp.cpu().data.numpy())
    print ('accuracy on test: ', acc/n)
    #print (np.mean(test_loss))
        





