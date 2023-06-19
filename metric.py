import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def item(self):
        return self.avg



class MAD(nn.Module):
    """ Root mean square error """

    def __init__(self):
        super(MAD, self).__init__()

    def forward(self, im1, im2, mask=None):
        assert im1.shape == im2.shape, 'input shape mismatch'

        rmse = torch.abs(im1 - im2)
        if mask is not None:
            rmse = (rmse * mask).flatten(1).sum(-1)
            rmse = rmse / mask.flatten(1).sum(-1)
        rmse = rmse.mean()
        return rmse




class RMSE(nn.Module):
    """ Root mean square error """

    def __init__(self):
        super(RMSE, self).__init__()

    def forward(self, im1, im2, mask=None):
        assert im1.shape == im2.shape, 'input shape mismatch'

        rmse = (im1 - im2).pow(2)
        if mask is not None:
            rmse = (rmse * mask).flatten(1).sum(-1)
            rmse = rmse / mask.flatten(1).sum(-1)
        rmse = rmse.mean().sqrt()
        return rmse


class PSNR(nn.Module):
    """ Peak signal-to-noise ratio """

    def __init__(self):
        super(PSNR, self).__init__()

    def forward(self, im1, im2):
        assert im1.shape == im2.shape, 'input shape mismatch'

        bs = im1.size(0)
        mse = (im1 - im2).pow(2).mean() + 1e-8
        psnr = -10 * mse.log10()
        return psnr


class SSIM(nn.Module):
    """ Structural similarity index measure """
    
    def __init__(self, n_channels=3, kernel_size=11, sigma=1.5):
        super(SSIM, self).__init__()

        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.sigma = sigma

        tics = torch.arange(kernel_size)[:, None]
        kernel = torch.exp(
            -(tics - kernel_size // 2).pow(2) / (2 * sigma ** 2)
        )
        kernel = kernel / kernel.sum()
        kernel = torch.mm(kernel, kernel.t())
        kernel = kernel.expand(n_channels, 1, -1, -1).contiguous()
        self.register_buffer('kernel', kernel, persistent=False)

        self.c1 = 0.01 ** 2
        self.c2 = 0.03 ** 2

    def forward(self, im1, im2):
        assert im1.shape == im2.shape, 'input shape mismatch'
        if im1.size(1) == 1:
            im1 = im1.expand(-1, self.n_channels, -1, -1)
            im2 = im2.expand(-1, self.n_channels, -1, -1)
        assert im1.size(1) == self.n_channels, 'number of channels mismatch'

        mu1 = F.conv2d(im1, self.kernel, groups=self.n_channels)
        mu2 = F.conv2d(im2, self.kernel, groups=self.n_channels)
        
        im1_sq, im2_sq = im1.pow(2), im2.pow(2)
        mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
        im1_im2, mu1_mu2 = im1 * im2, mu1 * mu2

        sigma1_sq = F.conv2d(im1_sq, self.kernel, groups=self.n_channels) - mu1_sq
        sigma2_sq = F.conv2d(im2_sq, self.kernel, groups=self.n_channels) - mu2_sq
        sigma12 = F.conv2d(im1_im2, self.kernel, groups=self.n_channels) - mu1_mu2

        tmp1 = (2 * mu1_mu2 + self.c1) * (2 * sigma12 + self.c2)
        tmp2 = (mu1_sq + mu2_sq + self.c1) * (sigma1_sq + sigma2_sq + self.c2)
        ssim_map = tmp1 / tmp2
        ssim = ssim_map.mean()
        return ssim

def crop_to_cal(img_gt,pred):
    int_mask = img_gt >0
    int_mask = int_mask.float()
    box = int_mask[0,0,:,:]  
    H = box.shape[0]
    # import ipdb
    # ipdb.set_trace()
    x1=y1=x2=y2=0
    for i in range(H):
        if box[i,:].mean().item()!=0:
            y1=i
            break
    for i in range(H):
        # print(i)
        if box[H-1-i,:].mean().item()!=0:
            y2=H -1 -i
            break
    for i in range(H):
        if box[:,i].mean().item()!=0:
            x1=i
            break
    for i in range(H):
        if box[:,H-1-i].mean().item()!=0:
            x2= H -1 -i
            break
    # print(x1,y1,x2,y2)
    
    padding = 40 
    if y1-padding>=0: y1-=padding
    if y2+padding<=H: y2+=padding
    if x1-padding>=0: x1-=padding
    if x2+padding<=H: x2+=padding

    eval_mask = np.zeros((H,H))
    eval_mask[y1:y2, x1:x2] = 1
    eval_mask = eval_mask.astype(np.bool)
    eval_h = y2-y1 
    eval_w = x2-x1
    img_gt_np = img_gt.cpu().numpy()   
    pred_np = pred.cpu().numpy() 
    # print(img_gt_np.shape,pred_np.shape)       
    disp_np = np.reshape(pred_np[:,:,eval_mask], [pred_np.shape[0],pred_np.shape[1],eval_h, eval_w])
    img_gt_np = np.reshape(img_gt_np[:,:,eval_mask], [img_gt_np.shape[0],img_gt_np.shape[1],eval_h, eval_w])
    
    # import cv2
    # show = img_gt_np[0].transpose(1,2,0)
    # cv2.imwrite('/data1/yueli/code/nlost_cvpr2023_10_1/crop_gt.png', (show/np.max(show)*255))
    # show = disp_np[0].transpose(1,2,0)
    # cv2.imwrite('/data1/yueli/code/nlost_cvpr2023_10_1/crop_int.png', (show/np.max(show)*255))

    disp_tensor = torch.from_numpy(disp_np)
    gt_tensor = torch.from_numpy(img_gt_np)

    return gt_tensor, disp_tensor