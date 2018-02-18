import torch
import torch.nn as nn
import torch.nn.parallel
import torchvision
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

class _netG(nn.Module):
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz + 1, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input, label):
        label = label.view(-1,1)
        input = input.view(-1,self.nz)
        
        #print (type(input), type(label), type(input.data), type(label.data), label.size(), input.size())

        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            #input.data = torch.stack([input.data,label], 0)
            input  = torch.cat([input,label],1).view(-1,self.nz + 1, 1, 1)
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            #label = Variable(label)
            input  = torch.cat([input,label],1).view(-1,self.nz + 1, 1, 1)
            
            output = self.main(input)
        return output
    
    
class _netD(nn.Module):
    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.ndf = ndf
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
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
            nn.Conv2d(ndf * 8, ndf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
            #n.Linear(ndf*8+1, 
            #nn.Sigmoid()
        )

    def forward(self, input, label):
        #print (type(input), type(label), type(input.data), type(label.data), label.size(), input.size())
        #print (self.ngpu)
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            bs = output.size(0)
            output = output.view(bs,-1)
            label = label.view(-1,1)
         
            lin = nn.Linear(self.ndf * 8 + 1,1).cuda()
            output = torch.cat([output, label],1).view(bs, -1)
          
            #output = lin(output)
            output =  nn.parallel.data_parallel(lin, output, range(self.ngpu))
            #sig = nn.Sigmoid()
            #output =  nn.parallel.data_parallel(sig, output, range(self.ngpu))
            
        else:
            
            output = self.main(input)
            bs = output.size(0)
            output = output.view(bs,-1)
            label = label.view(-1,1)
            #print (label.size(), output.size())
            #rint (input.size())
            lin = nn.Linear(self.ndf * 8 + 1,1).cuda()
            output = torch.cat([output, label],1).view(bs, -1)
            #print (output.size())
            #print (output.size())
            #print (self.ndf*8)
            output = lin(output)
            #sig = nn.Sigmoid()
            #output = sig(output)
        output = output.mean(0)

        return output.view(-1, 1).squeeze(1)


class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            self.fc1 = nn.Linear(size_out_1*size_out_2*n_filters + nrand, 1024)
            self.fc2 = nn.Linear(1024, 1024)
            self.fc3 = nn.Linear(1024, 3*J)

            self.conv1 = nn.Conv2d(1, n_filters, n_conv)
            self.conv2 = nn.Conv2d(n_filters,n_filters,n_conv)
            self.conv3 = nn.Conv2d(n_filters, n_filters, n_conv)

            self.mp1 = nn.MaxPool2d(n_pool)
            self.mp2 = nn.MaxPool2d(n_pool)

        def forward(self, x, z):
            h = F.relu(self.conv1(x))
            h = self.mp1(h)
            h = F.relu(self.conv2(h))
            h = self.mp2(h)
            h = F.relu(self.conv2(h))
            data = h.data
            data.resize_(x.size(0), size_out_1*size_out_2*n_filters)
            h = Variable(data)

            h_extended = torch.cat([h, z],1)
            h = h_extended
            h = F.relu(self.fc1(h))
            h = F.relu(self.fc2(h))
            h = F.relu(self.fc3(h))
            return h

class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.conv1 = nn.Conv2d(1, n_filters, n_conv)
            self.conv2 = nn.Conv2d(n_filters,n_filters,n_conv)
            self.conv3 = nn.Conv2d(n_filters, n_filters, n_conv)
            self.mp1 = nn.MaxPool2d(n_pool)
            self.mp2 = nn.MaxPool2d(n_pool)

            self.fc1 = nn.Linear(size_out_1*size_out_2*n_filters + 3 * J, 200)
            self.fc2 = nn.Linear(200, 200)
            self.fc3 = nn.Linear(200, 1)
        def forward(self, x, z):
            h = F.relu(self.conv1(x))
            h = self.mp1(h)
            h = F.relu(self.conv2(h))
            h = self.mp2(h)
            h = F.relu(self.conv2(h))

            data = h.data
            data.resize_(h.size(0), size_out_1*size_out_2*n_filters)
            h = Variable(data)
            h_extended = torch.cat([h, z],1)

            h = F.relu(self.fc1(h_extended))
            h = F.relu(self.fc2(h))
            h = F.sigmoid(self.fc3(h))
            return h
