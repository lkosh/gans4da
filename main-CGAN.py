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
from torch import autograd

import models.dcgan_v2_cgan as dcganv2
import models.dcgan as dcgan
import models.mlp as mlp
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nc', type=int, default=3, help='input image channels')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
parser.add_argument('--noBN', action='store_true', help='use batchnorm or not (only for DCGAN)')
parser.add_argument('--mlp_G', action='store_true', help='use MLP for G')
parser.add_argument('--mlp_D', action='store_true', help='use MLP for D')
parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
opt = parser.parse_args()
EPS = 1e-12

print(opt)



def calc_gradient_penalty(netD, real_data, fake_data, labels):
    """
    calculate gradient penalty
    """
    #print real_data.size()
    alpha = torch.rand(opt.batchSize, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if opt.cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if opt.cuda:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates, labels)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if opt.cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lamb
    return gradient_penalty



if opt.experiment is None:
    opt.experiment = 'samples'
os.system('mkdir {0}'.format(opt.experiment))

opt.manualSeed = random.randint(1, 10000) # fix seed

# opt.lambda - parameter of interpolation
opt.lamb = 0.5
###############3
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    )
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                                  
                                         shuffle=True, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = int(opt.nc)
n_extra_layers = int(opt.n_extra_layers)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

if opt.noBN:
    netG = dcgan.DCGAN_G_nobn(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)
elif opt.mlp_G:
    netG = mlp.MLP_G(opt.imageSize, nz, nc, ngf, ngpu)
else:
    #netG = dcgan.DCGAN_G(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)
    netG = dcganv2._netG(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)

netG.apply(weights_init)
if opt.netG != '': # load checkpoint if needed
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

if opt.mlp_D:
    netD = mlp.MLP_D(opt.imageSize, nz, nc, ndf, ngpu)
else:
    #netD = dcgan.DCGAN_D(opt.imageSize, nz, nc, ndf, ngpu, n_extra_layers)
    netD = dcganv2._netD(opt.imageSize, nz, nc, ndf, ngpu, n_extra_layers)

    netD.apply(weights_init)

if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
label = torch.LongTensor(opt.batchSize, 1)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
y_onehot = torch.LongTensor(opt.batchSize, 43)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
#fixed_labels = torch.FloatTensor(np.arange(43))
one = torch.FloatTensor([1])
mone = one * -1

label_compare = torch.FloatTensor(opt.batchSize)


if opt.cuda:
    netD.cuda()
    netG.cuda()
    input = input.cuda()
    label = label.cuda()
    one, mone = one.cuda(), mone.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
    y_onehot = y_onehot.cuda()
    label_compare = label_compare.cuda()

# setup optimizer
if opt.adam:
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))
else:
    optimizerD = optim.RMSprop(netD.parameters(), lr = opt.lrD)
    optimizerG = optim.RMSprop(netG.parameters(), lr = opt.lrG)
Diters = 1
gen_iterations = 0
criterion = nn.BCELoss()
criterion_MSE = nn.MSELoss()

for epoch in range(opt.niter):
    data_iter = iter(dataloader)
    i = 0
    
    while i < len(dataloader):
        ############################
        # (1) Update D network
        ###########################
        # for p in netD.parameters(): # reset requires_grad
        #     p.requires_grad = True # they are set to False below in netG update



        # train the discriminator Diters times
       
        

            # clamp parameters to a cube
            #for p in netD.parameters():
            #    p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

        data = data_iter.next()
        netD.zero_grad()

        i += 1

        # train with real
        real_cpu, labels_real_cpu = data
        
        #print (labels_real_cpu)
        labels_real_cpu =labels_real_cpu.long()
        batch_size = real_cpu.size(0)

        
                                      
        if opt.cuda:
            real_cpu = real_cpu.cuda()
            labels_real_cpu = labels_real_cpu.cuda()

        input.resize_as_(real_cpu).copy_(real_cpu)
        label.resize_as_(labels_real_cpu).copy_(labels_real_cpu)
        inputv = Variable(input)

        #labelv = Variable (label.float())
        label.resize_(label.shape[0], 1)
        tmp = torch.LongTensor(label.shape[0], 43).cuda()
        #print (label.shape, tmp.shape)
        y_onehot.resize_as_(tmp)
        tmp.zero_()
        tmp.scatter_(1, label, 1)
        print (tmp.sum())
        #print (tmp.shape)
        labelv = Variable(tmp.float())
            #print (label.unsqueeze(1).size())
        try:
            y_onehot.zero_().scatter_(1, label.unsqueeze(1), 1)
            #print (1)
        except:
            y_onehot.zero_().scatter_(1, label, 1)
            #print (2)
        
        
        y_onehot = torch.unsqueeze(torch.unsqueeze(y_onehot, 2),3).expand(label.shape[0], 43, 64,64)
        y_onehotv = Variable(y_onehot.float())
       
        errD_real = netD(inputv, y_onehotv)
        label_compare.resize_(batch_size).fill_(1 ) # use smooth label for discriminator
        label_comparev = Variable(label_compare)
        
        errD_real = criterion(errD_real, label_comparev)
        errD_real.backward()
        #print (errD_real.size())

        batchsize = min(opt.batchSize, input.size(0))
        # train with fake
        noise.resize_(batchsize, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise, volatile = True) # totally freeze netG
        fake = Variable(netG(noisev, labelv).data)
        inputv = fake
        #inputv = Variable(inputv)
        #print ("inputv", inputv.size())
        errD_fake = netD(inputv, y_onehotv)
        #print (errD_fake)
        
        
        label_compare.resize_(batch_size).fill_(0 ) # use smooth label for discriminator
        label_comparev = Variable(label_compare)
        errD_fake = criterion(errD_fake, label_comparev)
        
        #print (errD_fake.volatile, label_comparev.volatile, inputv.volatile, y_onehotv.volatile)
        errD_fake.backward()
        errD = errD_real + errD_fake
        #print (errD)
        #print ("errD", errD)
        #errD.backward()

        optimizerD.step()
      
        ############################
        # (2) Update G network
        ###########################
        
#         for p in netD.parameters():
#             p.requires_grad = False # to avoid computation


        # in case our last batch was the tail batch of the dataloader,
        # make sure we feed a full batch of noise
        for k in range(3):
            netG.zero_grad()

            noise.resize_(batchsize, nz, 1, 1).normal_(0, 1)
            noisev = Variable(noise)
            #labelv = Variable(torch.FloatTensor(np.repeat(np.arange(42),opt.batchSize/42 + 1)[:opt.batchSize])).cuda()
            fake = netG(noisev, labelv)
            inputv = fake
            #print (noisev.size(), labelv.size())
            label_compare.resize_(batch_size).fill_(1 ) # use smooth label for discriminator
            label_comparev = Variable(label_compare)
            #fake = netG(noisev, labelv)
            fake_labels = netD(inputv, y_onehotv)
            #print (fake_labels)
            errG = criterion(fake_labels,label_comparev )

            errG.backward()
            optimizerG.step()

        gen_iterations += 1

        print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
            % (epoch, opt.niter, i, len(dataloader), gen_iterations,
            errD.data.cpu().numpy()[0], errG.data.cpu().numpy()[0], errD_real.data.cpu().numpy()[0], errD_fake.data.cpu().numpy()[0]))
        if gen_iterations % 500 == 0:
            real_cpu = real_cpu.mul(0.5).add(0.5)
            vutils.save_image(real_cpu, '{0}/real_samples.png'.format(opt.experiment))
            fake = netG(Variable(fixed_noise, volatile=True), labelv)
            fake.data = fake.data.mul(0.5).add(0.5)
            vutils.save_image(fake.data, '{0}/fake_samples_{1}.png'.format(opt.experiment, gen_iterations))

    # do checkpointing
    torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(opt.experiment, epoch))
    torch.save(netD.state_dict(), '{0}/netD_epoch_{1}.pth'.format(opt.experiment, epoch))
