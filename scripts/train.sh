#!/usr/bin/env bash
job_name="goatee"
attr_name="Goatee"
    python -u train.py --n_blocks 3 --ngf 16 --ndf 64 --batch_size 24 --img_size 256\
    --sel_attrs $attr_name --name $job_name --gpu_ids 0 --use_lsgan --display_freq 50 \
    --lambda_gan_feat 5 --lambda_cls 2e-1 --print_freq 20 --lambda_flow_reg 1 --lambda_mask 1e-1
