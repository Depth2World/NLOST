# The train file for network
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

from util.LFEDataset import LFEDataset,NLOSDataset
from util.ParseArgs import parse_args
from util.SaveChkp import save_checkpoint
import util.SetDistTrain as utils
from pro.Train import train
import logging
from models import nlost


cudnn.benchmark = True

def main():
    
    # parse arguments
    opt = parse_args()
    # set distribution training (add args.distributed = True)
    utils.init_distributed_mode(opt)
    device = torch.device(opt.device)

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
    

    logging.info("+++++++++++++++++++++++++++++++++++++++++++")
    logging.info(opt)
    logging.info("Current main process GPUs: {}".format((opt.loc_rank)))
    logging.info("Number of available GPUs: {} {}".format(torch.cuda.device_count(), \
        torch.cuda.get_device_name(torch.cuda.current_device())))
    logging.info("Number of Encoder-Decoders: {}".format(opt.num_coders))
    logging.info("+++++++++++++++++++++++++++++++++++++++++++")
    
    # load data
    logging.info("Loading training and validation data...")
    # build dataset
    # fix the seed in dataset building for reproducibility
    seed = opt.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    folder_path = ['/data/yueli/dataset/bike']
    shineness = [0]
    logging.info(folder_path[0])
    train_data = LFEDataset(folder_path,shineness,True,1,512,256,1,[0.05,2],128,0,'gray')
    val_data = LFEDataset(folder_path,shineness,False,1,512,256,1,[0.05,2],128,0,'gray')
    
    # root_path = '/data/yueli/dataset/zip/NLOS_bike_allviews_color_processed'
    # logging.info(root_path)
    # train_data = NLOSDataset(root_path,True,1,512,256,2,[0.05,0.03],128,0,'gray')
    # val_data = NLOSDataset(root_path,False,1,512,256,2,[0.05,0.03],128,0,'gray')

    if opt.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        # if we want to use the repeated augmentation for data, use RASampler() in Twins-main
        train_sampler = DistributedSampler(train_data,num_replicas=num_tasks, rank=global_rank, shuffle=True)
        # for validation dataset, we sample it in sequential to keep the same val results among different validation
        val_sampler = SequentialSampler(val_data)
    else:
        train_sampler = RandomSampler(train_data)
        val_sampler = SequentialSampler(val_data)

    train_loader = DataLoader(train_data, sampler=train_sampler,batch_size=opt.bacth_size, num_workers=opt.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_data, sampler=val_sampler,batch_size=opt.bacth_size, num_workers=opt.num_workers, pin_memory=True)
    logging.info("Load training and validation data complete!")
    logging.info("+++++++++++++++++++++++++++++++++++++++++++")

    # build network and move it multi-GPU
    logging.info("Constructing Models...")
    model = nlost.NLOST(ch_in=1)
    
    model.to(device)
    logging.info(model)
    # ParamCounter(model)
    if opt.distributed:
        model = DDP(model, device_ids=[opt.loc_rank],find_unused_parameters=True)
        logging.info("Models constructed complete! Paralleled on {} GPUs".format(torch.cuda.device_count()))
    else:
        logging.info("Models constructed complete on SINGLE GPU!")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Total Numbers of parameters are: {}".format(num_params/1e6))
    logging.info("+++++++++++++++++++++++++++++++++++++++++++")
    
    # build optimizer
    params = filter(lambda p: p.requires_grad, model.parameters())
    if opt.opter == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=opt.lr_rate, weight_decay=opt.weit_decay)
    else:
        optimizer = torch.optim.Adam(params, lr=opt.lr_rate)

    n_iter = 0
    start_epoch = 1
    items = ["ALL", "int", "dep"]
    train_loss = {items[0]: [], items[1]: [], items[2]: []}
    val_loss = {items[0]: [], items[1]: [], items[2]: []}
    logWriter = SummaryWriter(opt.model_dir + "/")
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
                for k in ckpt_dict.keys():
                    model_dict.update({k[7:]: ckpt_dict[k]})
                model.load_state_dict(model_dict)
                logging.info("Loaded and update model states!")
            except KeyError as ke:
                logging.info("No model states found!")
                sys.exit("NO MODEL STATES")

            # load optimizer state
            for g in optimizer.param_groups:
                g["lr"] = opt.lr_rate
            logging.info("Loaded learning rate!")
               
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

        if opt.distributed:
            train_loader.sampler.set_epoch(epoch)
        
        model, optimizer, n_iter, train_loss, val_loss, logWriter,train_metrics,val_metrics = \
            train(model, train_loader, val_loader, optimizer, \
                epoch, n_iter, train_loss, val_loss, opt, logWriter)

        log_str = 'Epoch_Train_{} | '.format(epoch)
        for k in train_metrics:
            log_str += '{:s} {:.5f} | '.format(k, train_metrics[k].item())
        if utils.is_main_process():
            logging.info(log_str)       

        log_str = 'Epoch_Test_{} | '.format(epoch)
        for k in val_metrics:
            log_str += '{:s} {:.5f} | '.format(k, val_metrics[k].item())
        if utils.is_main_process():
            logging.info(log_str)

     
        for g in optimizer.param_groups:
            if (epoch<=30 or epoch>=40):
                g["lr"] *= 1.0
            else:
                g["lr"] *= .95
        
        # save checkpoint every epoch (not the dict file to save)
        if utils.is_main_process():
            save_checkpoint(n_iter, epoch, model, optimizer,\
                file_path=opt.model_dir+"/epoch_{}_{}_END.pth".format(epoch, n_iter))

        logging.info("End of epoch: {}. Checkpoint saved!".format(epoch))


if __name__=="__main__":
    logging.info("+++++++++++++++++++++++++++++++++++++++++++")
    logging.info("Sleeping...")
    time.sleep(3600*0)
    logging.info("Wake UP")
    logging.info("+++++++++++++++++++++++++++++++++++++++++++")
    logging.info("Execuating code...")
    main()



