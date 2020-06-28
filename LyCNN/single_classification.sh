#!/bin/bash
#SBATCH --time=41:06:66 # walltime, abbreviated by -t                           
#SBATCH --mem=110G                                                              
#SBATCH --job-name="shmerp"                                                     
#SBATCH -o model_shallow.out-%j # name of the stdout, using the job number (%j) and the first node (%N)    
#SBATCH -e model_shallow.err-%j # name of the stderr, using the job number (%j) and the first node (%N) 
#SBATCH --gres=gpu:1

model="$1"

#    %%  ___example_run___  %%
#
#    sbatch (model) (multi_crop) (crop_per_case) (kernels) (expansion) (depth) (aug_norm) (script) (server) 
#
#    sbatch ./train_log/model 3 20 16 12 40 0 multiThreadNet.py 1 1


# note: multi_crop==0 for single crop (non-multiple).

echo 1: model "$model", 2: multi_crop "$2", 3: crop_per_case "$3", 4: kernels "$4", 5: expansion "$5", 6: depth "$6", 7: aug_randhe "$7", 8:script "$8", 9:py "$9" , 10:server ${10} , 11:unknown_dir ${11}
#source ~/.bashrc

#medusa==1  fsm==0
if [ $9 -eq 1 ]
then
    PY='/home/sci/samlev/anaconda3/envs/tf2/bin/python3.5'
else
    PY='python3.4'
fi
#medusa==1 fsm==0 hard assign gpu == 0
if [ $9 -eq 1 ]
then
    GPU='--num_gpu' #/home/sci/samlev/anaconda3/envs/tf2/bin/python3.5'
    N=1
else
    GPU='--gpu'
    N=1
fi

# if fsm change env 
if [ ${10} -eq 0 ] 
then
    source ~/.bashfsm
    __conda_setup="$('/home/sci/samlev/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
    if [ $? -eq 0 ]; then
	eval "$__conda_setup"
    else
	if [ -f "/home/sci/samlev/anaconda3/etc/profile.d/conda.sh" ]; then
            . "/home/sci/samlev/anaconda3/etc/profile.d/conda.sh"
	else
            export PATH="/home/sci/samlev/anaconda3/bin:$PATH"
	fi
    fi
    unset __conda_setup
    conda activate tf2
fi

$PY $8 --tot test $GPU $N --mp 1 --nccl 1 --batch_size 500 --model_name ep_134_unkown_A --load "$model" --unknown_dir ${11} --depth $6 --class_weights 1,1 --crop_per_case $3 --multi_crop $2 --kernels $4 --expansion $5 --aug_norm 0 --aug_randhe $7 

#$PY $8 --tot test $GPU $N --mp 1 --nccl 1 --batch_size 500 --model_name ep_134_unkown_A --load "$model" --unknown_dir HQ_BL --depth $6 --class_weights 1,1 --crop_per_case $3 --multi_crop $2 --kernels $4 --expansion $5 --aug_norm 0 --aug_randhe $7

#$PY $8 --tot test $GPU $N --mp 1 --nccl 1 --batch_size 500 --model_name ep_134_unkown_A --load "$model" --unknown_dir A --depth $6 --class_weights 1,1 --crop_per_case $3 --multi_crop $2 --kernels $4 --expansion $5 --aug_norm 0 --aug_randhe $7

#$PY $8 --tot test $GPU $N --mp 1 --nccl 1 --batch_size 500 --model_name ep_134_unkown_A --load "$model" --unknown_dir B --depth $6 --class_weights 1,1 --crop_per_case $3 --multi_crop $2 --kernels $4 --expansion $5 --aug_norm 0 --aug_randhe $7

#$PY $8 --tot test $GPU $N --mp 1 --nccl 1 --batch_size 500 --model_name ep_134_unkown_A --load "$model" --unknown_dir C --depth $6 --class_weights 1,1 --crop_per_case $3 --multi_crop $2 --kernels $4 --expansion $5 --aug_norm 0 --aug_randhe $7

#$PY $8 --tot test $GPU $N --mp 1 --nccl 1 --batch_size 500 --model_name ep_134_unkown_A --load "$model" --unknown_dir D --depth $6 --class_weights 1,1 --crop_per_case $3 --multi_crop $2 --kernels $4 --expansion $5 --aug_norm 0 --aug_randhe $7

#$PY $8 --tot test $GPU $N --mp 1 --nccl 1 --batch_size 500 --model_name ep_134_unkown_A --load "$model" --unknown_dir E --depth $6 --class_weights 1,1 --crop_per_case $3 --multi_crop $2 --kernels $4 --expansion $5 --aug_norm 0 --aug_randhe $7

#$PY $8 --tot test $GPU $N --mp 1 --nccl 1 --batch_size 500 --model_name ep_134_unkown_A --load "$model" --unknown_dir F --depth $6 --class_weights 1,1 --crop_per_case $3 --multi_crop $2 --kernels $4 --expansion $5 --aug_norm 0 --aug_randhe $7

#$PY $8 --tot test $GPU $N --mp 1 --nccl 1 --batch_size 500 --model_name ep_134_unkown_A --load "$model" --unknown_dir G --depth $6 --class_weights 1,1 --crop_per_case $3 --multi_crop $2 --kernels $4 --expansion $5 --aug_norm 0 --aug_randhe $7

#$PY $8 --tot test $GPU $N --mp 1 --nccl 1 --batch_size 500 --model_name ep_134_unkown_A --load "$model" --unknown_dir H --depth $6 --class_weights 1,1 --crop_per_case $3 --multi_crop $2 --kernels $4 --expansion $5 --aug_norm 0 --aug_randhe $7

#$PY $8 --tot test $GPU $N --mp 1 --nccl 1 --batch_size 500 --model_name ep_134_unkown_A --load "$model" --unknown_dir I --depth $6 --class_weights 1,1 --crop_per_case $3 --multi_crop $2 --kernels $4 --expansion $5 --aug_norm 0 --aug_randhe $7

#$PY $8 --tot test $GPU $N --mp 1 --nccl 1 --batch_size 500 --model_name ep_134_unkown_A --load "$model" --unknown_dir K --depth $6 --class_weights 1,1 --crop_per_case $3 --multi_crop $2 --kernels $4 --expansion $5 --aug_norm 0 --aug_randhe $7

#$PY $8 --tot test $GPU $N --mp 1 --nccl 1 --batch_size 500 --model_name ep_134_unkown_A --load "$model" --unknown_dir L --depth $6 --class_weights 1,1 --crop_per_case $3 --multi_crop $2 --kernels $4 --expansion $5 --aug_norm 0 --aug_randhe $7

#$PY $8 --tot test $GPU $N --mp 1 --nccl 1 --batch_size 500 --model_name ep_134_unkown_A --load "$model" --unknown_dir M --depth $6 --class_weights 1,1 --crop_per_case $3 --multi_crop $2 --kernels $4 --expansion $5 --aug_norm 0 --aug_randhe $7

#$PY $8 --tot test $GPU $N --mp 1 --nccl 1 --batch_size 500 --model_name ep_134_unkown_A --load "$model" --unknown_dir N --depth $6 --class_weights 1,1 --crop_per_case $3 --multi_crop $2 --kernels $4 --expansion $5 --aug_norm 0 --aug_randhe $7

#$PY $8 --tot test $GPU $N --mp 1 --nccl 1 --batch_size 500 --model_name ep_134_unkown_A --load "$model" --unknown_dir O --depth $6 --class_weights 1,1 --crop_per_case $3 --multi_crop $2 --kernels $4 --expansion $5 --aug_norm 0 --aug_randhe $7

#$PY $8 --tot test $GPU $N --mp 1 --nccl 1 --batch_size 500 --model_name ep_134_unkown_A --load "$model" --unknown_dir P --depth $6 --class_weights 1,1 --crop_per_case $3 --multi_crop $2 --kernels $4 --expansion $5 --aug_norm 0 --aug_randhe $7

#$PY $8 --tot test $GPU $N --mp 1 --nccl 1 --batch_size 500 --model_name ep_134_unkown_A --load "$model" --unknown_dir Q --depth $6 --class_weights 1,1 --crop_per_case $3 --multi_crop $2 --kernels $4 --expansion $5 --aug_norm 0 --aug_randhe $7

#$PY $8 --tot test $GPU $N --mp 1 --nccl 1 --batch_size 500 --model_name ep_134_unkown_A --load "$model" --unknown_dir R --depth $6 --class_weights 1,1 --crop_per_case $3 --multi_crop $2 --kernels $4 --expansion $5 --aug_norm 0 --aug_randhe $7

#$PY $8 --tot test $GPU $N --mp 1 --nccl 1 --batch_size 500 --model_name ep_134_unkown_A --load "$model" --unknown_dir SP --depth $6 --class_weights 1,1 --crop_per_case $3 --multi_crop $2 --kernels $4 --expansion $5 --aug_norm $7

#$PY $8 --tot test $GPU $N --mp 1 --nccl 1 --batch_size 500 --model_name ep_134_unkown_A --load "$model" --unknown_dir J1 --depth $6 --class_weights 1,1 --crop_per_case $3 --multi_crop $2 --kernels $4 --expansion $5 --aug_norm 0 --aug_randhe $7

#$PY $8 --tot test $GPU $N --mp 1 --nccl 1 --batch_size 500 --model_name ep_134_unkown_A --load "$model" --unknown_dir J2 --depth $6 --class_weights 1,1 --crop_per_case $3 --multi_crop $2 --kernels $4 --expansion $5 --aug_norm 0 --aug_randhe $7
