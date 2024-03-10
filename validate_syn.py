import os
import sys
import time
import numpy as np
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler, RandomSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import scipy.io as scio
import cv2
from tqdm import tqdm
from util.LFEDataset import LFEDataset
from metric import RMSE, PSNR, SSIM ,MAD, crop_to_cal,MAD,cal_psnr
import torch.nn.functional as F

cudnn.benchmark = True
from models import nlost
import logging

def main(args):
    
    # baseline   
    model = nlost.NLOST(ch_in=1, num_coders=1,spatial=128,tlen=256,bin_len=0.01,target_size=args.target_size)
    model.cuda()
    model = torch.nn.DataParallel(model)
    model_path = args.pretrained_model
    if model_path is not None:
        checkpoint = torch.load(model_path, map_location="cpu")
        model_dict = model.state_dict()
        ckpt_dict = checkpoint['state_dict']
        model_dict.update(ckpt_dict)
        #for k in ckpt_dict.keys():
        #    model_dict.update({k[7:]: ckpt_dict[k]})
        model.load_state_dict(model_dict)
        print('Loaded', model_path)
    else:
        print('Loading Failed', model_path)
    
    # start training 
    print("Start eval...")
    folder_path = [args.syn_data_path]
    shineness = [0]
    val_data = LFEDataset(root=folder_path,  # dataset root directory
                          shineness=shineness,
                          for_train=False,
                          ds=1,  # temporal down-sampling factor
                          clip=512,  # time range of histograms
                          size=256,  # measurement size (unit: px)
                          scale=1,  # scaling factor (float or float tuple)
                          background=[0.05, 2],  # background noise rate (float or float tuple)
                          target_size=args.target_size,  # target image size (unit: px)
                          target_noise=0.01,  # standard deviation of target image noise
                          color='gray')  # color channel(s) of target image
    
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    out_path = args.output_path + '/syn/'
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
    if logging.root: del logging.root.handlers[:]
    logging.basicConfig(
      level=logging.INFO,
      handlers=[
        logging.FileHandler(args.output_path + '/crop_syn_evluation.log' ),
        logging.StreamHandler()
      ],
      format='%(relativeCreated)d:%(levelname)s:%(process)d-%(processName)s: %(message)s'
    )
    
    niter = 1
    l_psnr = []
    l_ssim = []
    l_mad = []
    l_im = []
    l_rmse = []

    cal_ssim = SSIM().cuda()
    cal_mad = MAD().cuda()
    mse = nn.MSELoss()
    with torch.no_grad():
        model.eval()
        for sample in tqdm(val_loader):
            # if niter==10:break;
            M_wnoise = sample["ds_meas"].cuda()
            dep_gt = sample["dep_gt"].cuda()
            img_gt = sample["img_gt"].cuda()
            mask = dep_gt > 0
            mask = mask.float()
            ###### predict ######
            _,pred,depth_pred = model(M_wnoise)

            pred = (pred + 1) / 2
            depth_pred = (depth_pred + 1) / 2

            
            ### raw size
            # box_gt, box_pred = img_gt,pred
            # box_dep_gt, box_dep_pred = dep_gt,depth_pred
            
            ### crop the central region for evluation
            box_gt, box_pred = crop_to_cal(img_gt,pred)
            box_dep_gt, box_dep_pred = crop_to_cal(dep_gt,depth_pred)
            
            box_gt, box_pred = box_gt.cuda(), box_pred.cuda()
            box_dep_gt, box_dep_pred = box_dep_gt.cuda(), box_dep_pred.cuda()

            im_psnr = cal_psnr(box_pred, box_gt)
            l_psnr.append(im_psnr.item())
            im_ssim = cal_ssim(box_pred, box_gt)
            l_ssim.append(im_ssim.item())
            dep_mad = cal_mad(box_dep_pred, box_dep_gt)
            l_mad.append(dep_mad.item())
            dep_rmse = torch.sqrt(mse(box_dep_pred, box_dep_gt))
            l_rmse.append(dep_rmse.item())
            

            ###### store ######
            front_view = pred[0].cpu().numpy().transpose(1,2,0)
            depth_view = depth_pred[0].cpu().numpy().transpose(1,2,0)
            view_gt = img_gt[0].cpu().numpy().transpose(1,2,0)
            cv2.imwrite(out_path + f'{niter}_int.png', (front_view/np.max(front_view)*255))
            cv2.imwrite(out_path + f'{niter}_gt.png', (view_gt/np.max(view_gt)*255))
            cv2.imwrite(out_path + f'{niter}_dep.png', (depth_view*255))
            niter += 1
                
        logging.info("img_ssim: %f" % (float(sum(l_ssim))/float(len(l_ssim))))
        logging.info("dep_mad: %f" % (float(sum(l_mad))/float(len(l_mad))))
        logging.info("img_psnr: %f" % (float(sum(l_psnr))/float(len(l_psnr))))
        logging.info("dep_rmse: %f" % (float(sum(l_rmse))/float(len(l_rmse))))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--syn_data_path", type=str, default=" ",help="Path to the syn dataset.")
    parser.add_argument("--target_size", type=int, default=" ",help="The spatial resolution of the input transient, i.e., 256 or 128")
    parser.add_argument("--output_path", type=str, default=" ",help="Path to output.")  
    parser.add_argument("--pretrained_model", type=str, default=" ",help="Prtrained Model Path.")  
    args = parser.parse_args()

    return args

def test():
    args = parse_args()
    main(args)

if __name__=="__main__":
    test()
    
