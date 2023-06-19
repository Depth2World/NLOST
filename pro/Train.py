# The train function
import sys
sys.path.append("../util/")
import numpy as np 
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import scipy.io as scio
from pro.Validate import validate,test_on_align_fk, test_on_align_xu 
from util.SaveChkp import save_checkpoint
import util.SetDistTrain as utils
from pro.Loss import criterion_KL, criterion_L2, criterion_TV
#import cv2
import time
import logging

from metric import RMSE, PSNR, SSIM, AverageMeter
cudnn.benchmark = True
lsmx = torch.nn.LogSoftmax(dim=1)
smx = torch.nn.Softmax(dim=1)
dtype = torch.cuda.FloatTensor
items = ["ALL", "int", "dep"]

metric_list = ['rmse', 'psnr', 'ssim']
train_metrics = {k: AverageMeter() for k in metric_list}
val_metrics = {k: AverageMeter() for k in metric_list}

def train(model, train_loader, val_loader, optimer, epoch, n_iter,
            train_loss, val_loss, params, logWriter):
    device = torch.device(params.device)
    rmse = RMSE().to(device)
    psnr = PSNR().to(device)
    ssim = SSIM().to(device)
    for sample in tqdm(train_loader):
        # configure model state
        model.train()
        # load data and train the network
        M_mea = sample["M_nos"].type(dtype)
        dep_gt = sample["dep_gt"].type(dtype)
        img_gt = sample["img_gt"].type(dtype)
        # mask = img_gt > 0

        M_mea_re, inten_re,dep_re = model(M_mea)

        #t2 = time.time()
        #cost_time = t2 -t1
        #print('Forward time',cost_time)
        #total_cost += cost_time
        #if n_iter%200==0:
        #    print('Mean Forward time',total_cost/200)
        #    total_cost = 0
        loss_l2 = criterion_L2(inten_re, img_gt)
        loss_l2_dep = criterion_L2(dep_re, dep_gt)

        loss =  loss_l2 + loss_l2_dep
        # loss =  loss_l2
        optimer.zero_grad()
        loss.backward()
        optimer.step()
        n_iter += 1
       
        metric_dict = {
            'rmse': 0,
            'psnr': 0,
            'ssim': 0,
        }
        for k in metric_list:
            train_metrics[k].update(metric_dict[k])

        if utils.is_main_process():
            logWriter.add_scalar("loss_train/all", loss, n_iter)
            logWriter.add_scalar("loss_train/l2", loss_l2, n_iter)
            logWriter.add_scalar("loss_train/l2_dep", loss_l2_dep, n_iter)

            for k in metric_list:
                logWriter.add_scalars(k, {'train': train_metrics[k].item()}, n_iter)
            train_loss[items[0]].append(loss.data.cpu().numpy())
            train_loss[items[1]].append(loss_l2.data.cpu().numpy())
            train_loss[items[2]].append(loss_l2_dep.data.cpu().numpy())

        if n_iter % params.num_save == 0:
        # if n_iter % 1 == 0:
            logging.info("Sart validation...")
            with torch.no_grad():
                logWriter = test_on_align_fk(model, n_iter, logWriter, params)
                logWriter = test_on_align_xu(model, n_iter, logWriter, params)
                val_loss, logWriter, val_metric = validate(model, val_loader, n_iter, val_loss, params, logWriter,val_metrics)
                log_str = 'test {} | '.format(n_iter)
                for k in metric_list:
                    log_str += '{:s} {:.5f} | '.format(k, val_metric[k].item())
                logging.info(log_str)

            if utils.is_main_process():
                scio.savemat(file_name=params.model_dir+"/train_loss.mat", mdict=train_loss)
                scio.savemat(file_name=params.model_dir+"/val_loss.mat", mdict=val_loss)
                # save model states
                logging.info("Validation complete!")
                save_checkpoint(n_iter, epoch, model, optimer,
                file_path=params.model_dir+"/epoch_{}_{}.pth".format(epoch, n_iter))
                logging.info("Checkpoint saved!")
    
    return model, optimer, n_iter, train_loss, val_loss, logWriter,train_metrics,val_metrics

