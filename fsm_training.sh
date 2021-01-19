#!/bin/bash

source ~/.bashrc
conda activate fsm_tftpidx

python ./LyCNN/multiThreadDenseNNet_Lymphoma.py --data_dir ./data --gpu 1 --tot train --lr_init 6.5e-5 --lr_0 2.5e-6 --drop_0 40 --lr_1 2.5e-7 --drop_1 60 --drop_2 75 --max_epoch 400 --frac_res 0.80 --gpu_frac 0.8 --mp 2 --image_size 224 --batch_size 32 --class_weights 1,1 --kernel_size 3 --nccl 1 --drop_out 0.0 --drop_pattern 0 --aug_he 0 --aug_randnorm 0.0 --aug_randhe 0.7 --model_name Memory_Loging --multi_crop 3 --crop_per_case 250 --kernels 12 --expansion 12 --depth 24 --aug_norm 0
