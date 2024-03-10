import os
import sys
import time
import numpy as np
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from datetime import datetime
import scipy.io as scio
from tqdm import tqdm
import cv2
from models.utils_pytorch import phasor_1_10
from models.utils_pytorch import fk_1_10
from models.utils_pytorch import lct_1_10

def main():
    # baseline   
    spatial = 128
    temp_bin = 512
    bin_len = ( 512 // temp_bin ) * 0.0096 
    model = phasor_1_10.phasor(spatial=spatial, crop=temp_bin, bin_len=bin_len, wall_size = 2, sampling_coeff=2.0, cycles=5,dnum=1)
    # model = fk_1_10.lct_fk(spatial=spatial, crop=temp_bin, bin_len=bin_len,wall_size = 2, dnum=1)
    # model = lct_1_10.lct(spatial=spatial, crop=temp_bin, bin_len=bin_len,method='bp',dnum=1)
    # model = lct_1_10.lct(spatial=spatial, crop=temp_bin, bin_len=bin_len,method='lct',dnum=1)
    model.cuda()
    model = torch.nn.DataParallel(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total Numbers of parameters are: {}".format(num_params))
    print("+++++++++++++++++++++++++++++++++++++++++++")
    print("Start eval...")
    
    rw_path  = '/data/yueli/dataset/NLOS_RW/cvpr2023_data/'
    out_path = '/data/yueli/code/nlost_cvpr2023_master_modified/pretrain/tra_alg/rsd/' 
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
    all_file = []
    files = os.listdir(rw_path)
    for fi in files:
        fi_d = os.path.join(rw_path, fi)
        all_file.append(fi_d)
    ims = []
    for i in range(len(all_file)): 
        transient_data = scio.loadmat(all_file[i])
        transient_data = transient_data['data']  
        transient_data = transient_data.transpose([1,0,2])
        M_wnoise = np.asarray(transient_data).astype(np.float32).reshape([1,128,128,-1]) 

        c, h, w,t = M_wnoise.shape
        ds = t//temp_bin
        M_wnoise = M_wnoise.reshape(c, h, w, t // ds, ds)
        M_wnoise = M_wnoise.sum(axis=4)
        M_wnoise = np.ascontiguousarray(M_wnoise)
        M_wnoise = np.transpose(M_wnoise, (0, 3, 1, 2))  
        M_mea = torch.from_numpy(M_wnoise)  
        print(M_mea.size())
        M_mea = M_mea[None]
        with torch.no_grad():
            model.eval()
            re = model(M_mea,[0,0,0],[temp_bin,temp_bin,temp_bin])
            volumn_MxNxN = re.detach().cpu().numpy()[0, -1]
            # zdim = volumn_MxNxN.shape[0] * 100 // 128
            # volumn_MxNxN = volumn_MxNxN[:zdim]
            
            volumn_MxNxN[volumn_MxNxN < 0] = 0
            front_view = np.max(volumn_MxNxN, axis=0)
            front_view = front_view / np.max(front_view)
            front_view = (front_view*255).astype(np.uint8)
            
            depth_view = np.argmax(volumn_MxNxN, axis=0)
            depth_view = depth_view.astype(np.float32)/np.max(depth_view)
            depth_view = (depth_view*255).astype(np.uint8)
            
            cv2.imwrite(out_path   + files[i][:-4] + '_int.png',front_view)
            cv2.imwrite(out_path   + files[i][:-4] + '_dep.png',depth_view)
            # scio.savemat(out_path  + f'{i}.mat',{'mea':volumn_MxNxN})
    
    
if __name__=="__main__":
    main()




