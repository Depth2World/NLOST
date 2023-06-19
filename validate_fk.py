 # The train file for network
# Based on pytorch 1.8
# Use the docker: nlos_trans:1.8 in Ubuntu
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

from util.SpadDataset import SpadDataset
from util.SetRandomSeed import set_seed, worker_init
from util.ParseArgs import parse_args
from util.SaveChkp import save_checkpoint
from util.MakeDataList import makelist
import util.SetDistTrain as utils
from pro.Train import train
from tqdm import tqdm
 
from models import model_tst_NLOS_NBlks
from models import model_ddfn_CGNL
from models import sup_model
from models import model_unet

cudnn.benchmark = True
dtype = torch.cuda.FloatTensor
lsmx = torch.nn.LogSoftmax(dim=1)
smx = torch.nn.Softmax(dim=1)
from pro.Loss import criterion_KL, criterion_L2
from skimage.metrics import structural_similarity as ssim
from models import model_tst_NLOS_NBlks,model_tst_NLOS_NBlks_phy

def main():
    
    # baseline   
    # model = model_tst_NLOS_NBlks_phy.TSTIntemodelNBlk_LCGC_128_512_v2(ch_in=1)
    model = model_tst_NLOS_NBlks_phy.TSTIntemodelNBlk_LCGC_128_512_v9_14(ch_in=1)

    model_path = '/data/yueli/output/CVPR2023_nlosp/nlosp_v9_14_fk_LFEbike_Resize_color2gray_2022_1010/epoch_9_6000.pth'
    model.cuda()
    model = torch.nn.DataParallel(model)
    # print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total Numbers of parameters are: {}".format(num_params))
    print("+++++++++++++++++++++++++++++++++++++++++++")
    
    checkpoint = torch.load(model_path, map_location="cpu")
    model_dict = model.state_dict()
    ckpt_dict = checkpoint['state_dict']
    model_dict.update(ckpt_dict)
    #for k in ckpt_dict.keys():
    #    model_dict.update({k[7:]: ckpt_dict[k]})
    model.load_state_dict(model_dict)
    print("Start eval...")
    # rw_path  = '/data/yueli/dataset/align_fk_256_512'
    # rw_path  = '/data/yueli/dataset/xu_512'
    rw_path  = '/data/yueli/dataset/xu_new_data128'

    #rw_path = '/data1/pengjy/NLOS/Data_simu/real_world/align_fk_all'
    out_path = '/data/yueli/output/nlost_v9_14_up_gray/niter6000/fk/'
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
    all_file = []
    files = os.listdir(rw_path)
    for fi in files:
        fi_d = os.path.join(rw_path, fi)
        all_file.append(fi_d)
    for i in range(len(all_file)): 

        transient_data = scio.loadmat(all_file[i])
        # transient_data = transient_data['data'].transpose([2,1,0])  #sig final_meas measlr
        # transient_data = transient_data['final_meas']  #sig final_meas measlr
        # M_wnoise = np.asarray(transient_data).astype(np.float32).reshape([1,256,256,-1])   # 1, 1, 64, 64,2048  8ps
        # transient_data = transient_data[:,::-1,:]
        # M_wnoise = np.asarray(transient_data).astype(np.float32).reshape([1,512,512,-1])
        # M_wnoise = M_wnoise[:,::2,:,:] + M_wnoise[:,1::2,:,:]
        # M_wnoise = M_wnoise[:,:,::2,:] + M_wnoise[:,:,1::2,:] 
        # M_wnoise = M_wnoise[:,::2,:,:] + M_wnoise[:,1::2,:,:]
        # M_wnoise = M_wnoise[:,:,::2,:] + M_wnoise[:,:,1::2,:]

        transient_data = transient_data['data'].transpose([1,0,2])   #sig final_meas measlr
        M_wnoise = np.asarray(transient_data).astype(np.float32).reshape([1,128,128,-1])


        M_wnoise = np.ascontiguousarray(M_wnoise)
        M_wnoise = np.transpose(M_wnoise, (0, 3, 1, 2))  
        M_mea = torch.from_numpy(M_wnoise[None])  
        # print(M_mea.size())
        with torch.no_grad():
            model.eval()
            vlo_re, im_re,dep_re = model(M_mea)
            front_view = im_re.detach().cpu().numpy()[0, 0]
            front_dep = dep_re.detach().cpu().numpy()[0, 0]
            vlo = vlo_re.detach().cpu().numpy()[0, 0]
            import cv2
            cv2.imwrite(out_path + f'/{i}.png', (front_view / np.max(front_view))*255)
            cv2.imwrite(out_path + f'/{i}d.png', (front_dep / np.max(front_dep))*255)
            # # import matplotlib.pyplot as plt
            # plt.imshow(dep_np.squeeze())
            # plt.savefig(out_path + files[i] + '_pred_256.png')
            scio.savemat(out_path + f'/{i}.mat',{'pred_mea':vlo})
if __name__=="__main__":
    print("+++++++++++++++++++++++++++++++++++++++++++")
    print("Sleeping...")
    time.sleep(3600*0)
    print("Wake UP")
    print("+++++++++++++++++++++++++++++++++++++++++++")
    print("Execuating code...")
    main()




