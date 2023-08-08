# The validation function
import sys
sys.path.append("../util/")
import numpy as np 
import torch
from tqdm import tqdm
import util.SetDistTrain as utils
from pro.Loss import criterion_KL, criterion_L2, criterion_TV
import scipy.io as scio
import os 
import logging
from models import nlost
import cv2

from metric import RMSE, PSNR, SSIM, AverageMeter
lsmx = torch.nn.LogSoftmax(dim=1)
smx = torch.nn.Softmax(dim=1)
dtype = torch.cuda.FloatTensor
items = ["ALL", "int", "dep"]
metric_list = ['rmse', 'psnr', 'ssim']

def validate(model, val_loader, n_iter, val_loss, params, logWriter,val_metrics):
    
    device = torch.device(params.device)
    rmse = RMSE().to(device)
    psnr = PSNR().to(device)
    ssim = SSIM().to(device)
    model.eval()
    l_all = []
    l_l2 = []
    l_l2_dep = []

    for sample in tqdm(val_loader):
        M_mea = sample["M_nos"].type(dtype)
        dep_gt = sample["dep_gt"].type(dtype)
        img_gt = sample["img_gt"].type(dtype)
        mask = img_gt > 0
        # mask = mask.float()
        M_mea_re, int_re,dep_re = model(M_mea)

        # M_mea_re, int_re = model(M_mea)
        loss_l2 = criterion_L2(int_re, img_gt).data.cpu().numpy()
        
        loss_l2_dep = criterion_L2(dep_re, dep_gt).data.cpu().numpy()

        loss = loss_l2  + loss_l2_dep
        # loss = loss_l2  
        l_all.append(loss)
        l_l2.append(loss_l2)
        l_l2_dep.append(loss_l2_dep)

        pred = torch.clamp(int_re.detach(), 0, 1)
        target = torch.clamp(img_gt, 0, 1)
        metric_dict = {
            'rmse': rmse(pred, target).cpu(),
            'psnr': psnr(pred, target).cpu(),
            'ssim': ssim(pred, target).cpu(),
        }
        for k in metric_list:
            val_metrics[k].update(metric_dict[k].item())
    # log the val losses
    if utils.is_main_process():
        logWriter.add_scalar("loss_val/all", np.mean(l_all), n_iter)
        logWriter.add_scalar("loss_val/l2", np.mean(l_l2), n_iter)
        logWriter.add_scalar("loss_val/l2_dep", np.mean(l_l2_dep), n_iter)

        logWriter.add_images("inten_rec", int_re.clamp(0,1), n_iter,dataformats="NCHW")
        logWriter.add_images("inten_gt", img_gt, n_iter,dataformats="NCHW")
        logWriter.add_images("dep_rec", dep_re.clamp(0,1), n_iter,dataformats="NCHW")
        logWriter.add_images("dep_gt", dep_gt, n_iter,dataformats="NCHW")

        for k in metric_list:
            logWriter.add_scalars(k, {'test': val_metrics[k].item()}, n_iter)
        val_loss[items[0]].append(np.mean(l_all))
        val_loss[items[1]].append(np.mean(l_l2))
        val_loss[items[2]].append(np.mean(l_l2_dep))
    return val_loss, logWriter, val_metrics



def test_on_align_fk(model, n_iter, logWriter, params):
    model_dict = model.state_dict()
    test_model = nlost.NLOST(ch_in=1, num_coders=1,spatial=128,tlen=256,bin_len=0.0096)
    test_model = torch.nn.DataParallel(test_model,[0])
    test_model.load_state_dict(model_dict)
    out_path = params.model_dir + '/test_on_fk/'
    if not os.path.exists(out_path):
            os.makedirs(out_path, exist_ok=True) 

    rw_path = '/data/yueli/dataset/align_fk_256_512'
    files = os.listdir(rw_path)
    all_file = []
    for fi in files:
        fi_d = os.path.join(rw_path, fi)
        all_file.append(fi_d)
    for i in range(len(all_file)): 
        transient_data = scio.loadmat(all_file[i])['final_meas']
        M_wnoise = np.asarray(transient_data).astype(np.float32).reshape([1,256,256,-1])   # 1, 1, 256, 256, 512  32ps
        M_wnoise = M_wnoise[:,::2,:,:] + M_wnoise[:,1::2,:,:]
        M_wnoise = M_wnoise[:,:,::2,:] + M_wnoise[:,:,1::2,:]
        
        M_wnoise = np.ascontiguousarray(M_wnoise)
        M_wnoise = np.transpose(M_wnoise, (0, 3, 1, 2))  
        M_mea = torch.from_numpy(M_wnoise[None])  

        if utils.is_main_process():
            with torch.no_grad():
                test_model.eval()
                _,im_re,dep_re = test_model(M_mea)
                # logWriter.add_images('fk/inten'+str(files[i]), im_re/torch.max(im_re), n_iter,dataformats="NCHW")
                front_view = im_re.detach().cpu().numpy()[0, 0]
                # logWriter.add_images('fk/dep'+str(files[i]), dep_re/torch.max(dep_re), n_iter,dataformats="NCHW")
                # front_dep = dep_re.detach().cpu().numpy()[0, 0]
                # os.makedirs(params.model_dir + f'/test_on_fk/{n_iter}/', exist_ok=True)
                cv2.imwrite(out_path + f'n_iter{n_iter}_{i}.png', (front_view / np.max(front_view))*255)

    return logWriter
    


def test_on_align_xu(model, n_iter, logWriter, params):
    model_dict = model.state_dict()
    test_model = nlost.NLOST(ch_in=1, num_coders=1,spatial=128,tlen=256,bin_len=0.0096)
    test_model = torch.nn.DataParallel(test_model,[0])
    test_model.load_state_dict(model_dict)
    rw_path  = '/data/yueli/dataset/cvpr2023_data'

    out_path = params.model_dir + '/test_on_our/'
    if not os.path.exists(out_path):
            os.makedirs(out_path, exist_ok=True) 

    files = os.listdir(rw_path)
    all_file = []
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

        if utils.is_main_process():
            with torch.no_grad():
                test_model.eval()
                _,im_re,dep_re = test_model(M_mea)
                # logWriter.add_images('fk/inten'+str(files[i]), im_re/torch.max(im_re), n_iter,dataformats="NCHW")
                front_view = im_re.detach().cpu().numpy()[0, 0]
                # logWriter.add_images('fk/dep'+str(files[i]), dep_re/torch.max(dep_re), n_iter,dataformats="NCHW")
                # front_dep = dep_re.detach().cpu().numpy()[0, 0]
                # os.makedirs(params.model_dir + f'/test_on_xu_new/{n_iter}/', exist_ok=True)
                cv2.imwrite(out_path + f'n_iter{n_iter}_{i}.png', (front_view / np.max(front_view))*255)
    return logWriter
    
