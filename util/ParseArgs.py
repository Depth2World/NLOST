# The parse argments
import os
from pickle import TRUE
import sys
import argparse
import configparser
from configparser import ConfigParser, ExtendedInterpolation
from datetime import datetime


def get_args_parser():
    # build the arg_parser
    parser = argparse.ArgumentParser("Transformer", add_help=False)
    # set path params
    parser.add_argument("--param_store", type=bool, default=True)
    parser.add_argument("--model_name", type=str, default="github")  
    parser.add_argument("--model_dir", type=str, default="/data/yueli/output/CVPR2023_nlosp/")
    parser.add_argument("--seed", type=int, default=3407, help="seed for data loading")
    parser.add_argument("--resmue", type=bool, default=False)
    parser.add_argument("--resmod_dir", type=str, default='')  
    parser.add_argument("--resm_tran", type=str, default="")
    parser.add_argument("--resm_test", type=str, default="")
    # set model params
    parser.add_argument("--bacth_size", type=int, default=4)
    parser.add_argument("--down_scale", type=int, default=1)
    parser.add_argument("--num_epoch", type=int, default=51)
    parser.add_argument("--num_save", type=int, default=300) #800
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--num_coders", type=int, default=1)
    parser.add_argument("--lr_rate", type=float, default=1e-4)
    parser.add_argument("--weit_decay", type=int, default=1e-4)
    parser.add_argument("--loss_weit", type=float, default=0.)
    parser.add_argument("--grad_clip", type=int, default=0.1, help="gradient clipping max norm")
    parser.add_argument("--noise_idx", type=int, default=1)
    parser.add_argument("--num_head", type=int, default=8)
    parser.add_argument("--drop_attn", type=float, default=0.)
    parser.add_argument("--drop_proj", type=float, default=0.)
    parser.add_argument("--drop_path", type=float, default=0.)
    # set optimization params
    parser.add_argument("--opter", type=str, default="adamw")
    parser.add_argument("--epo_warm", type=int, default=5)
    parser.add_argument("--epo_cool", type=int, default=5)
    # set distributed training params
    parser.add_argument("--dp_gpus", type=str, default="0,1,2,3")# 
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--world_size", type=int, default=1, help="number of total machines")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--dist_url", type=str, default="env://", help="url used to setup the distributed training")

    return parser


def parse_args():
    parser = argparse.ArgumentParser("Transformer", parents=[get_args_parser()])
    args = parser.parse_args()
    # update some params
    today = datetime.today()
    if args.resmue==True:
        args.model_dir += args.model_name +"_resume_"+ str(today.year)+"_"+str(today.month)+str(today.day)
    else:
        args.model_dir += args.model_name +"_"+ str(today.year)+"_"+str(today.month)+str(today.day)

    # mkdirs if necessary
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir, exist_ok=True)
    # save args to files
    if args.param_store:
        args_dict = args.__dict__
        config_bk_pth = args.model_dir + "/config_bk.txt"
        with open(config_bk_pth, "w") as cbk_pth:
            cbk_pth.writelines("------------------Start------------------"+ "\n")
            for key, valus in args_dict.items():
                cbk_pth.writelines(key + ": " + str(valus) + "\n")
            cbk_pth.writelines("------------------End------------------"+ "\n")
            
        print("Config file load complete! \nNew file saved to {}".format(config_bk_pth))

    return args

