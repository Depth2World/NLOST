#!/bin/bash
echo "sleeping..."
sleep 1s
echo "Code running..."
python -m torch.distributed.launch --nproc_per_node=2 --use_env main_DDP.py \
--bacth_size 1 \
--model_dir /data/yueli/CVPR2023_output/CVPR2023_nlosp/ # output path
