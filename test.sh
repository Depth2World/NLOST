
###  A sample test on fk data with spatial size 256 
python validate_fk.py \
--fk_data_path /data/yueli/dataset/NLOS_RW/align_fk_256_512 \
--target_size 256 \
--output_path /data/yueli/code/NLOST/pretrain/256_fk \
--pretrained_model /data/yueli/code/NLOST/pretrain/size256.pth

###  A sample test on our data  with spatial size 256 
python validate_ours.py \
--fk_data_path /data/yueli/dataset/NLOS_RW/cvpr2023_data \
--target_size 256 \
--output_path /data/yueli/code/NLOST/pretrain/256_ours \
--pretrained_model /data/yueli/code/NLOST/pretrain/size256.pth



###  A sample test on fk data with spatial size 128 
python validate_fk.py \
--fk_data_path /data/yueli/dataset/NLOS_RW/align_fk_256_512 \
--target_size 128 \
--output_path /data/yueli/code/NLOST/pretrain/128_fk \
--pretrained_model /data/yueli/code/NLOST/pretrain/size128.pth

###  A sample test on our data  with spatial size 128 
python validate_ours.py \
--fk_data_path /data/yueli/dataset/NLOS_RW/cvpr2023_data \
--target_size 128 \
--output_path /data/yueli/code/NLOST/pretrain/128_ours \
--pretrained_model /data/yueli/code/NLOST/pretrain/size128.pth


###  A sample test on synthetic data with spatial size 128 
# python validate_syn.py \
# --syn_data_path /data/yueli/dataset/bike \
# --target_size 128 \
# --output_path /data/yueli/code/NLOST/pretrain/128_syn \
# --pretrained_model /data/yueli/code/NLOST/pretrain/size128.pth

###  A sample test on synthetic data with spatial size 256 
# python validate_syn.py \
# --syn_data_path /data/yueli/dataset/bike \
# --target_size 128 \
# --output_path /data/yueli/code/NLOST/pretrain/128_syn \
# --pretrained_model /data/yueli/code/NLOST/pretrain/size128.pth

