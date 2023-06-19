# The train file for network
# Based on pytorch 1.8
# Use the docker: nlos_trans:1.8 in Ubuntu
import os
import sys
import time
from cv2 import FlannBasedMatcher
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
import logging

from util.SpadDataset import SpadDataset,SparseCodingDataset
from util.LFEDataset import LFEDataset,NLOSDataset
from util.SetRandomSeed import set_seed, worker_init
from util.ParseArgs import parse_args
from util.SaveChkp import save_checkpoint
from util.MakeDataList import makelist
import util.SetDistTrain as utils
from pro.Train import train

from models import model_tst_NLOS_NBlks,model_tst_NLOS_NBlks_phy
from models import model_ddfn_CGNL
from models import sup_model
from models import model_unet

from metric import AverageMeter

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True

def main():
    
    # parse arguments
    opt = parse_args()
    print("+++++++++++++++++++++++++++++++++++++++++++")
    print(opt)
    # print("Current main process GPUs: {}".format((opt.loc_rank)))
    print("Number of available GPUs: {} {}".format(torch.cuda.device_count(), \
        torch.cuda.get_device_name(torch.cuda.current_device())))
    print("Number of Encoder-Decoders: {}".format(opt.num_coders))
    print('Setting gpu to {}'.format(opt.dp_gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(opt.dp_gpus)
    print("+++++++++++++++++++++++++++++++++++++++++++")
    # make train and val data list if necessary
    # if not os.path.exists(opt.tradt_dir) or not os.path.exists(opt.tesdt_dir):
    #     print("No train or validation data list found. \nMAKE THEM")
    #     makelist(opt)
    # else:
    #     print("Train and validation data list exist.")
    # set random seed for producibility 2022/4/8
    set_seed(opt)
    # load data
    
    if logging.root: del logging.root.handlers[:]
    logging.basicConfig(
      level=logging.INFO,
      handlers=[
        logging.FileHandler(opt.model_dir + '/train.log' ),
        logging.StreamHandler()
      ],
      format='%(relativeCreated)d:%(levelname)s:%(process)d-%(processName)s: %(message)s'
    )
    logging.info('='*80)
    logging.info(f'Start of experiment: {opt.model_name}')
    logging.info('='*80)
    
    logging.info("Loading training and validation data...")
    
    folder_path = ['/data/yueli/dataset/bike']
    shineness = [0]
    logging.info(folder_path[0])

    train_data = LFEDataset(folder_path,shineness,True,1,512,256,1,[0.05,2],128,0,'gray')
    val_data = LFEDataset(folder_path,shineness,False,1,512,256,1,[0.05,2],128,0,'gray')

    # root_path = '/data/yueli/dataset/zip/NLOS_bike_allviews_color_processed'
    # logging.info(root_path)
    # train_data = NLOSDataset(root_path,True,1,512,256,2,[0.05,0.2],128,0.01,'gray')
    # val_data = NLOSDataset(root_path,False,1,512,256,2,[0.05,0.2],128,0.01,'gray')

    train_loader = DataLoader(train_data, batch_size=opt.bacth_size, shuffle=True, num_workers=opt.num_workers, worker_init_fn=worker_init, pin_memory=True) # drop_last would also influence the performance
    val_loader = DataLoader(val_data, batch_size=opt.bacth_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
    logging.info("Load training and validation data complete!")
    logging.info("+++++++++++++++++++++++++++++++++++++++++++")
    # build network and move it multi-GPU
    torch.manual_seed(opt.seed)
    logging.info("Constructing Models...")
    # model = model_ddfn_CGNL.SP_baseline_v3(in_channels=1)
    model = model_tst_NLOS_NBlks_phy.TSTIntemodelNBlk_LCGC_128_512_v9_14(ch_in=1)
    
    model.cuda()
    model = torch.nn.DataParallel(model)
    logging.info(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Total Numbers of parameters are: {}".format(num_params))
    logging.info("+++++++++++++++++++++++++++++++++++++++++++")
    
    # build optimizer
    params = filter(lambda p: p.requires_grad, model.parameters())
    if opt.opter == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=opt.lr_rate, weight_decay=opt.weit_decay)
    else:
        optimizer = torch.optim.Adam(params, lr=opt.lr_rate)
    n_iter = 0
    start_epoch = 1
    items = ["ALL", "KL", "L2D", "L1I", "SSIMI", "TVD", "TVT"]
    train_loss = {items[0]: [], items[1]: [], items[2]: [], items[5]: []}
    val_loss = {items[0]: [], items[1]: [], items[2]: [], items[5]: []}

    logWriter = SummaryWriter(opt.model_dir + '/logs')
    L2Dmin = 1
    logging.info("Parameters initialized")
    logging.info("+++++++++++++++++++++++++++++++++++++++++++")
    if opt.resmue:
        if os.path.exists(opt.resmod_dir):
            logging.info("Loading checkpoint from {}".format(opt.resmod_dir))
            checkpoint = torch.load(opt.resmod_dir, map_location="cpu")
            # load start epoch
            try:
                start_epoch = checkpoint['epoch']
                logging.info("Loaded and update start epoch: {}".format(start_epoch))
            except KeyError as ke:
                start_epoch = 1
                logging.info("No epcoh info found in the checkpoint, start epoch from 1")
            # load iter number
            try:
                n_iter = checkpoint["n_iter"]
                logging.info("Loaded and update start iter: {}".format(n_iter))
            except KeyError as ke:
                n_iter = 0
                logging.info("No iter number found in the checkpoint, start iter from 0")
            # load learning rate
            try:
                opt.lr_rate = checkpoint["lr"]
            except KeyError as ke:
                logging.info("No learning rate info found in the checkpoint, use initial learning rate:")
            # load model params
            model_dict = model.state_dict()
            try:
                ckpt_dict = checkpoint['state_dict']
                model_dict.update(ckpt_dict)
                #for k in ckpt_dict.keys():
                    #model_dict.update({k[7:]: ckpt_dict[k]})
                    #model_dict.update({k[7:]: ckpt_dict[k]})
                model.load_state_dict(model_dict)
                logging.info("Loaded and update model states!")
            except KeyError as ke:
                logging.info("No model states found!")
                sys.exit("NO MODEL STATES")
            # simple model dict load methods (2021.12.6)
            # model_wo_ddp = model.module
            # model_wo_ddp.load_state_dict(checkpoint['state_dict'])
            # load optimizer state
            for g in optimizer.param_groups:
                g["lr"] = opt.lr_rate
            logging.info("Loaded learning rate!")
            # load mat files
            try:
                train_loss[items[0]] = np.squeeze(scio.loadmat(opt.resm_tran)[items[0]]).tolist()
                train_loss[items[1]] = np.squeeze(scio.loadmat(opt.resm_tran)[items[1]]).tolist()
                train_loss[items[2]] = np.squeeze(scio.loadmat(opt.resm_tran)[items[2]]).tolist()
                train_loss[items[5]] = np.squeeze(scio.loadmat(opt.resm_tran)[items[5]]).tolist()
                val_loss[items[0]] = np.squeeze(scio.loadmat(opt.resm_test)[items[0]]).tolist()
                val_loss[items[1]] = np.squeeze(scio.loadmat(opt.resm_test)[items[1]]).tolist()
                val_loss[items[2]] = np.squeeze(scio.loadmat(opt.resm_test)[items[2]]).tolist()
                val_loss[items[5]] = np.squeeze(scio.loadmat(opt.resm_test)[items[5]]).tolist()
                logging.info("Loaded and update train and val loss from assigned path!")
            except FileNotFoundError as fnf:
                logging.info("No train or val loss mat found.\nUse initial ZERO")
            
            logging.info("Checkpoint load complete!!!")

        else:
            logging.info("No checkPoint found at {}!!!".format(opt.resmod_dir))
            sys.exit("NO FOUND CHECKPOINT ERROR!")

    else:
        logging.info("Do not resume! Use initial params and train from scratch.")

    # start training 
    logging.info("Start training...")
    for epoch in range(start_epoch, opt.num_epoch):
        logging.info("Epoch: {}, LR: {}".format(epoch, optimizer.param_groups[0]["lr"]))
        
        model, optimizer, n_iter, train_loss, val_loss, logWriter,train_metrics,val_metrics = \
            train(model, train_loader, val_loader, optimizer, \
                epoch, n_iter, train_loss, val_loss, opt, logWriter)
        log_str = 'Epoch_Train_{} | '.format(epoch)
        for k in train_metrics:
            log_str += '{:s} {:.5f} | '.format(k, train_metrics[k].item())
        logging.info(log_str)

        log_str = 'Epoch_Test_{} | '.format(epoch)
        for k in val_metrics:
            log_str += '{:s} {:.5f} | '.format(k, val_metrics[k].item())
        logging.info(log_str)

        logging.info("==================>Train<==================")
        logging.info("{}: {}, {}: {}, {}: {}, {}: {}".format(\
            items[0], np.mean(train_loss[items[0]][-(len(train_data)-1):]),\
                items[1], np.mean(train_loss[items[1]][-(len(train_data)-1):]), \
                    items[2], np.mean(train_loss[items[2]][-(len(train_data)-1):]), \
                        items[5], np.mean(train_loss[items[5]][-(len(train_data)-1):])))
        logging.info("==================>Validation<==================")
        logging.info("{}: {}, {}: {}, {}: {}, {}: {}".format(\
            items[0], np.mean(val_loss[items[0]][-(len(train_data)//(opt.bacth_size*opt.num_save)-1):]), \
                items[1], np.mean(val_loss[items[1]][-(len(train_data)//(opt.bacth_size*opt.num_save)-1):]), \
                    items[2], np.mean(val_loss[items[2]][-(len(train_data)//(opt.bacth_size*opt.num_save)-1):]), \
                        items[5], np.mean(val_loss[items[5]][-(len(train_data)//(opt.bacth_size*opt.num_save)-1):])))

        # lr update 
        for g in optimizer.param_groups:
            if (epoch<=30 or epoch>=40):
                g["lr"] *= 1.0
            else:
                g["lr"] *= .95
        
        # save checkpoint every epoch (not the dict file to save)
        #save_checkpoint(n_iter, epoch, model, optimizer,\
        #    file_path=opt.model_dir+"/epoch_{}_{}_END.pth".format(epoch, n_iter))
        #print("End of epoch: {}. Checkpoint saved!".format(epoch))
        logging.info("End of epoch: {}. ".format(epoch))

if __name__=="__main__":
    logging.info("+++++++++++++++++++++++++++++++++++++++++++")
    logging.info("Sleeping...")
    time.sleep(3600*0)
    logging.info("Wake UP")
    logging.info("+++++++++++++++++++++++++++++++++++++++++++")
    logging.info("Execuating code...")
    main()



