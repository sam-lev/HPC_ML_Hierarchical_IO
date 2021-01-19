#!/bin/bash

source ~/.bashrc
conda activate slurmtftpidx

python ./LyCNN/multiThreadDenseNNet_Lymphoma.py --num_gpu 1 --tot train --lr_init 6.5e-3 --lr_0 2.5e-6 --drop_0 40 --lr_1 2.5e-7 --drop_1 60 --drop_2 75 --max_epoch 400 --frac_res 0.90 --gpu_frac 0.95 --mp 3 --image_size 224 --batch_size 32 --class_weights 1,1 --kernel_size 3 --nccl 1 --drop_out 0.0 --drop_pattern 0 --aug_he 0 --aug_randnorm 0.0 --aug_randhe 0.7 --model_name PIDX90V --multi_crop 3 --crop_per_case 250 --kernels 12 --expansion 12 --depth 22 --aug_norm 0 --data_dir /home/sci/samlev/Convolutional_Neural_Net_Lymphoma/data
