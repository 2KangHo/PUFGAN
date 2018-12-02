from __future__ import print_function
import argparse
import os
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

import torchvision.transforms as transforms
import torchvision.utils as vutils

from dataset import DatasetFromFolder
from networks import define_G, define_D, print_network


parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='data', help='path to dataset')
parser.add_argument('--workers', type=int,
                    help='number of data loading workers', default=16)
parser.add_argument('--batchSize', type=int,
                    default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64,
                    help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=256,
                    help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=500,
                    help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='learning rate, default=0.0005')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPUs to use')
parser.add_argument('--netG', default='',
                    help="path to netG (to continue training)")
parser.add_argument('--netD', default='',
                    help="path to netD (to continue training)")
parser.add_argument('--outf', default='result',
                    help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
    os.system('mkdir -p result/grid')
    os.system('mkdir -p result/single_0')
    os.system('mkdir -p result/single_32')
    os.system('mkdir -p result/model')
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.cuda:
    cudnn.benchmark = True
    torch.cuda.manual_seed(opt.manualSeed)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 1


print('===> Loading datasets')
dataset = DatasetFromFolder(opt.dataroot, opt.imageSize)
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

print('===> Building model')
netG = define_G(nc, nz, ngf, ngpu, device, opt.netG)
netD = define_D(nc, ndf, ngpu, device, opt.netD)

criterion = nn.BCELoss()

fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

print('---------- Networks initialized -------------')
print_network(netG)
print_network(netD)
print('-----------------------------------------------')

if opt.cuda:
    netD = netD.cuda()
    netG = netG.cuda()
    criterion = criterion.cuda()


for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real = data.to(device)
        batch_size = real.size(0)
        label = torch.full((batch_size,), real_label, device=device)

        output = netD(real)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        if i % 50 == 0:
            vutils.save_image(real,
                              '%s/grid/real_samples.png' % opt.outf,
                              normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(fake.detach(),
                              '%s/grid/fake_samples_epoch_%03d.png' % (
                                  opt.outf, epoch),
                              normalize=True)
            vutils.save_image(fake.detach()[0],
                              '%s/single_0/fake_samples_epoch_%03d_0.png' % (
                                  opt.outf, epoch),
                              normalize=True)
            vutils.save_image(fake.detach()[32],
                              '%s/single_32/fake_samples_epoch_%03d_32.png' % (
                                  opt.outf, epoch),
                              normalize=True)

    # do checkpointing
    if epoch % 20 == 0:
        torch.save(netG.state_dict(), '%s/model/netG_epoch_%d.pth' %
                   (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/model/netD_epoch_%d.pth' %
                   (opt.outf, epoch))
