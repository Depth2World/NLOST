import numpy as np 
from math import exp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models


def imgrad(img):
    img = torch.mean(img, 1, True)
    fx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv1.weight = nn.Parameter(weight)
    grad_x = conv1(img)

    fy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv2.weight = nn.Parameter(weight)
    grad_y = conv2(img)

    # grad = torch.sqrt(torch.pow(grad_x,2) + torch.pow(grad_y,2))
    return grad_y, grad_x


def imgrad_yx(img):
    N, C, _, _ = img.size()
    grad_y, grad_x = imgrad(img)
    # the computed grad's edge is useless, remove them
    grad_x = grad_x[:,:,2:-2,2:-2]
    grad_y = grad_y[:,:,2:-2,2:-2]
    out = torch.cat((torch.reshape(grad_y,(N,C,-1)), torch.reshape(grad_x,(N,C,-1))), dim=1)
    return out


class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()
    
    def forward(self, inpt, tar):
        grad_tar = imgrad_yx(tar)
        grad_inpt = imgrad_yx(inpt)
        loss = torch.sum(torch.mean(torch.abs(grad_tar - grad_inpt)))
        return loss


class L1_log(nn.Module):
    def __init__(self):
        super(L1_log, self).__init__()

    def forward(self, inpt, target):
        loss = nn.L1Loss()(torch.log(inpt), torch.log(target))
        return loss

        
###########################################
# the set of loss functions
criterion_GAN = nn.MSELoss()
criterion_KL = nn.KLDivLoss()

# inpt, target: [batch_size, 1, h, w]
criterion_L1 = nn.L1Loss()
criterion_L1log = L1_log()
criterion_grad = GradLoss()


def criterion_TV(inpt):
    return torch.sum(torch.abs(inpt[:, :, :, :-1] - inpt[:, :, :, 1:])) + \
           torch.sum(torch.abs(inpt[:, :, :-1, :] - inpt[:, :, 1:, :]))


def criterion_L2(est, gt):
    criterion = nn.MSELoss()
    # est should have grad
    return torch.sqrt(criterion(est, gt))


def criterion_KL_noise(est, gt):
    h = np.random.randint(32, size=2)
    w = np.random.randint(32, size=2)

    loss = criterion_KL(est[:,:,:, h[0], w[0]], gt[:, :, :, h[1], w[1]])

    return loss

