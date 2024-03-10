
###
# model_dir: output path 
# data_dir: the path of the synthetic data
# target_size: the spatial size of the input transient measurement, i.e., 256,128 
# nlostdata_path: the path of nlost real-world data
# fkdata_path：the path of fk real-world data

###
python -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port 29500 train.py \
--bacth_size 1 \
--model_dir /data/yueli/ \
--model_name nlost \
--data_dir /data/yueli/dataset/bike \
--nlost_datapath /data/yueli/dataset/NLOS_RW/cvpr2023_data \
--fk_datapath /data/yueli/dataset/NLOS_RW/align_fk_256_512 \
--target_size 256 
