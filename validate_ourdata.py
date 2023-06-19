# Based on pytorch 1.10
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
from models import nlost

def main():
    
    # baseline   
    model = nlost.NLOST(ch_in=1, num_coders=1,spatial=128,tlen=256,bin_len=0.0096)

    model_path = ''
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
    rw_path  = '/data/yueli/dataset/cvpr2023_data'

    out_path = ''
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
    all_file = []
    files = os.listdir(rw_path)
    for fi in files:
        fi_d = os.path.join(rw_path, fi)
        all_file.append(fi_d)
    for i in range(len(all_file)): 

        transient_data = scio.loadmat(all_file[i])
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




