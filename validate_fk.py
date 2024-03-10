import os
import sys
import time
import numpy as np
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import scipy.io as scio
import cv2
import util.SetDistTrain as utils
import cv2
import torch.nn.functional as F
cudnn.benchmark = True
from models import nlost
def main(args):
    
    # baseline   
    model = nlost.NLOST(ch_in=1, num_coders=1,spatial=128,tlen=256,bin_len=0.0096,target_size=args.target_size)

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

    print("Start eval...")
    all_file = []
    files = os.listdir(args.fk_data_path)
    for fi in files:
        fi_d = os.path.join(args.fk_data_path, fi)
        all_file.append(fi_d)

    out_path = args.output_path
    if not os.path.exists(out_path):
            os.makedirs(out_path, exist_ok=True)

    for i in range(len(all_file)): 
        transient_data = scio.loadmat(all_file[i])
        transient_data = transient_data['final_meas'] 
        M_wnoise = np.asarray(transient_data).astype(np.float32).reshape([1,256,256,-1])   # 1, 1, 64, 64,2048  8ps
        if args.target_size == 128:
            M_wnoise = M_wnoise[:,::2,:,:] + M_wnoise[:,1::2,:,:]
            M_wnoise = M_wnoise[:,:,::2,:] + M_wnoise[:,:,1::2,:]
        M_wnoise = np.ascontiguousarray(M_wnoise)
        M_wnoise = np.transpose(M_wnoise, (0, 3, 1, 2))  
        M_mea = torch.from_numpy(M_wnoise[None])  
        with torch.no_grad():
            model.eval()
            vlo_re, im_re,dep_re = model(M_mea)
            im_re = (im_re + 1) / 2
            dep_re = (dep_re + 1) / 2
            front_view = im_re.detach().cpu().numpy()[0, 0]
            front_dep = dep_re.detach().cpu().numpy()[0, 0]
            name = files[i][:-4]
            # vlo = vlo_re.detach().cpu().numpy()[0, 0]
            cv2.imwrite(out_path + f'/{name}_int.png', (front_view / np.max(front_view))*255)
            cv2.imwrite(out_path + f'/{name}_dep.png', (front_dep)*255)
            # scio.savemat(out_path + f'/{i}.mat',{'pred_mea':vlo})

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fk_data_path", type=str, default=" ",help="Path to the fk dataset.")
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
    




