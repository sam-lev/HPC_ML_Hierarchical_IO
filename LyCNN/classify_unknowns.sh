#!/bin/bash
#SBATCH --time=41:06:66 # walltime, abbreviated by -t                           
#SBATCH --mem=110G                                                              
#SBATCH --job-name="shmerp"                                                     
#SBATCH -o mem_idx_80_classify.out-%j # name of the stdout, using the job number (%j) and the first node (%N)
#SBATCH -e mem_idx_80_classify.err-%j # name of the stderr, using the job number (%j) and the first node (%N)
#SBATCH --gres=gpu:1

source ~/.bashrc

train_log="/home/sci/samlev/Convolutional_Neural_Net_Lymphoma/train_log"
model_set=$train_log/$1
opt_model="min-validation_error"
model=$model_set/$opt_model

#    %%  ___example_run___  %%
#
#    sbatch (model) (multi_crop) (crop_per_case) (kernels) (expansion) (depth) (aug_norm) (script) (server) 
#
#    sbatch ./train_log/model 4 20 16 12 40 0 multiThreadNet.py 1 1
# note: multi_crop==0 for single crop (non-multiple).
#multicrop=$3   #multicrop
#croppercase=$4 #crop per case
#kernels=$5     #kernels
#expansion=$6   #expansion
#depth=$7       #depth
#augrandhe=$8   #aug ramd he
#script=$9
#data_dir=$10

# presets for best model:
multicrop=0   #multicrop
croppercase=100000   #crop per case
kernels=12  #kernels
expansion=12  #expansion
depth=24  #depth
augrandhe=0   #aug ramd he
script='./LyCNN/multiThreadDenseNNet_Lymphoma.py'
datadir='/home/sci/samlev/Convolutional_Neural_Net_Lymphoma/data'
gpufrac=0.95

echo 1: model "$model", 3: multi_crop "$multicrop", 5: crop_per_case "$croppercase", 5: kernels "$kernels", 6: expansion "$expansion", 7: depth "$depth", 8: aug_randhe "$augrandhe", 9:script "$script" #, 9:py "$script" , 10:server ${10} ,

#medusa==1  fsm==0
if [ $2 -eq 1 ]
then
    conda activate slurmtftpidx
    PY='python'
    #/home/sci/samlev/anaconda3/envs/tf2/bin/python3.6'
else
    conda activate fsm_tftpidx
    PY='python'
fi

#medusa==1 fsm==0 hard assign gpu == 0
if [ $2 -eq 1 ]
then
    GPU='--num_gpu' #/home/sci/samlev/anaconda3/envs/tf2/bin/python3.6'
    N=1
else
    GPU='--gpu'
    N=1
fi

#$PY $script --tot test $GPU $N --mp 1 --nccl 1 --gpu_frac $gpufrac --batch_size 500 --model_name res80 --load "$model" --unknown_dir HQ_DLBCL --depth $depth --class_weights 1,1 --crop_per_case $croppercase --multi_crop $multicrop --kernels $kernels --expansion $expansion --aug_norm 0 --aug_randhe $augrandhe --data_dir $datadir

#$PY $script --tot test $GPU $N --mp 1 --nccl 1 --gpu_frac $gpufrac --batch_size 500 --model_name res80 --load "$model" --unknown_dir HQ_BL --depth $depth --class_weights 1,1 --crop_per_case $croppercase --multi_crop $multicrop --kernels $kernels --expansion $expansion --aug_norm 0 --aug_randhe $augrandhe --data_dir $datadir

$PY $script --tot test $GPU $N --mp 1 --nccl 1 --gpu_frac $gpufrac --batch_size 500 --model_name res80 --load "$model" --unknown_dir PCS-17-3002 --depth $depth --class_weights 1,1 --crop_per_case $croppercase --multi_crop $multicrop --kernels $kernels --expansion $expansion --aug_norm 0 --aug_randhe $augrandhe --data_dir $datadir

$PY $script --tot test $GPU $N --mp 1 --nccl 1 --gpu_frac $gpufrac --batch_size 500 --model_name res80 --load "$model" --unknown_dir HP-18-643 --depth $depth --class_weights 1,1 --crop_per_case $croppercase --multi_crop $multicrop --kernels $kernels --expansion $expansion --aug_norm 0 --aug_randhe $augrandhe --data_dir $datadir

$PY $script --tot test $GPU $N --mp 1 --nccl 1 --gpu_frac $gpufrac --batch_size 500 --model_name res80 --load "$model" --unknown_dir HP-16-1212_1 --depth $depth --class_weights 1,1 --crop_per_case $croppercase --multi_crop $multicrop --kernels $kernels --expansion $expansion --aug_norm 0 --aug_randhe $augrandhe --data_dir $datadir

$PY $script --tot test $GPU $N --mp 1 --nccl 1 --gpu_frac $gpufrac --batch_size 500 --model_name res80 --load "$model" --unknown_dir HP-14-453 --depth $depth --class_weights 1,1 --crop_per_case $croppercase --multi_crop $multicrop --kernels $kernels --expansion $expansion --aug_norm 0 --aug_randhe $augrandhe --data_dir $datadir

#$PY $script --tot test $GPU $N --mp 1 --nccl 1 --gpu_frac $gpufrac --batch_size 500 --model_name res80 --load "$model" --unknown_dir 5-HE --depth $depth --class_weights 1,1 --crop_per_case $croppercase --multi_crop $multicrop --kernels $kernels --expansion $expansion --aug_norm 0 --aug_randhe $augrandhe --data_dir $datadir

#$PY $script --tot test $GPU $N --mp 1 --nccl 1 --gpu_frac $gpufrac --batch_size 500 --model_name res80 --load "$model" --unknown_dir 7-HE --depth $depth --class_weights 1,1 --crop_per_case $croppercase --multi_crop $multicrop --kernels $kernels --expansion $expansion --aug_norm 0 --aug_randhe $augrandhe --data_dir $datadir

$PY $script --tot test $GPU $N --mp 1 --nccl 1 --gpu_frac $gpufrac --batch_size 500 --model_name res80 --load "$model" --unknown_dir SP-12-18833 --depth $depth --class_weights 1,1 --crop_per_case $croppercase --multi_crop $multicrop --kernels $kernels --expansion $expansion --aug_norm 0 --aug_randhe $augrandhe --data_dir $datadir

$PY $script --tot test $GPU $N --mp 1 --nccl 1 --gpu_frac $gpufrac --batch_size 500 --model_name res80 --load "$model" --unknown_dir 4-HE --depth $depth --class_weights 1,1 --crop_per_case $croppercase --multi_crop $multicrop --kernels $kernels --expansion $expansion --aug_norm 0 --aug_randhe $augrandhe --data_dir $datadir

$PY $script --tot test $GPU $N --mp 1 --nccl 1 --gpu_frac $gpufrac --batch_size 500 --model_name res80 --load "$model" --unknown_dir 6-HE --depth $depth --class_weights 1,1 --crop_per_case $croppercase --multi_crop $multicrop --kernels $kernels --expansion $expansion --aug_norm 0 --aug_randhe $augrandhe --data_dir $datadir

#$PY $script --tot test $GPU $N --mp 1 --nccl 1 --gpu_frac $gpufrac --batch_size 500 --model_name res80 --load "$model" --unknown_dir D --depth $depth --class_weights 1,1 --crop_per_case $croppercase --multi_crop $multicrop --kernels $kernels --expansion $expansion --aug_norm 0 --aug_randhe $augrandhe --data_dir $datadir

#$PY $script --tot test $GPU $N --mp 1 --nccl 1 --gpu_frac $gpufrac --batch_size 500 --model_name res80 --load "$model" --unknown_dir F --depth $depth --class_weights 1,1 --crop_per_case $croppercase --multi_crop $multicrop --kernels $kernels --expansion $expansion --aug_norm 0 --aug_randhe $augrandhe --data_dir $datadir

#$PY $script --tot test $GPU $N --mp 1 --nccl 1 --gpu_frac $gpufrac --batch_size 500 --model_name res80 --load "$model" --unknown_dir G --depth $depth --class_weights 1,1 --crop_per_case $croppercase --multi_crop $multicrop --kernels $kernels --expansion $expansion --aug_norm 0 --aug_randhe $augrandhe --data_dir $datadir

$PY $script --tot test $GPU $N --mp 1 --nccl 1 --gpu_frac $gpufrac --batch_size 500 --model_name res80 --load "$model" --unknown_dir H --depth $depth --class_weights 1,1 --crop_per_case $croppercase --multi_crop $multicrop --kernels $kernels --expansion $expansion --aug_norm 0 --aug_randhe $augrandhe --data_dir $datadir

#$PY $script --tot test $GPU $N --mp 1 --nccl 1 --gpu_frac $gpufrac --batch_size 500 --model_name res80 --load "$model" --unknown_dir N --depth $depth --class_weights 1,1 --crop_per_case $croppercase --multi_crop $multicrop --kernels $kernels --expansion $expansion --aug_norm 0 --aug_randhe $augrandhe --data_dir $datadir

#$PY $script --tot test $GPU $N --mp 1 --nccl 1 --gpu_frac $gpufrac --batch_size 500 --model_name res80 --load "$model" --unknown_dir O --depth $depth --class_weights 1,1 --crop_per_case $croppercase --multi_crop $multicrop --kernels $kernels --expansion $expansion --aug_norm 0 --aug_randhe $augrandhe --data_dir $datadir

#$PY $script --tot test $GPU $N --mp 1 --nccl 1 --gpu_frac $gpufrac --batch_size 500 --model_name res80 --load "$model" --unknown_dir Q --depth $depth --class_weights 1,1 --crop_per_case $croppercase --multi_crop $multicrop --kernels $kernels --expansion $expansion --aug_norm 0 --aug_randhe $augrandhe --data_dir $datadir

#$PY $script --tot test $GPU $N --mp 1 --nccl 1 --gpu_frac $gpufrac --batch_size 500 --model_name res80 --load "$model" --unknown_dir S --depth $depth --class_weights 1,1 --crop_per_case $croppercase --multi_crop $multicrop --kernels $kernels --expansion $expansion --aug_norm 0 --aug_randhe $augrandhe --data_dir $datadir

#$PY $script --tot test $GPU $N --mp 1 --nccl 1 --gpu_frac $gpufrac --batch_size 500 --model_name res80 --load "$model" --unknown_dir E --depth $depth --class_weights 1,1 --crop_per_case $croppercase --multi_crop $multicrop --kernels $kernels --expansion $expansion --aug_norm 0 --aug_randhe $augrandhe --data_dir $datadir



$PY $script --tot test $GPU $N --mp 1 --nccl 1 --gpu_frac $gpufrac --batch_size 500 --model_name res80 --load "$model" --unknown_dir SP-18-15882 --depth $depth --class_weights 1,1 --crop_per_case $croppercase --multi_crop $multicrop --kernels $kernels --expansion $expansion --aug_norm 0 --aug_randhe $augrandhe --data_dir $datadir

$PY $script --tot test $GPU $N --mp 1 --nccl 1 --gpu_frac $gpufrac --batch_size 500 --model_name res80 --load "$model" --unknown_dir SP-18-7695 --depth $depth --class_weights 1,1 --crop_per_case $croppercase --multi_crop $multicrop --kernels $kernels --expansion $expansion --aug_norm 0 --aug_randhe $augrandhe --data_dir $datadir

$PY $script --tot test $GPU $N --mp 1 --nccl 1 --gpu_frac $gpufrac --batch_size 500 --model_name res80 --load "$model" --unknown_dir SP-18-0022776 --depth $depth --class_weights 1,1 --crop_per_case $croppercase --multi_crop $multicrop --kernels $kernels --expansion $expansion --aug_norm 0 --aug_randhe $augrandhe --data_dir $datadir

$PY $script --tot test $GPU $N --mp 1 --nccl 1 --gpu_frac $gpufrac --batch_size 500 --model_name res80 --load "$model" --unknown_dir SP-18-26597 --depth $depth --class_weights 1,1 --crop_per_case $croppercase --multi_crop $multicrop --kernels $kernels --expansion $expansion --aug_norm 0 --aug_randhe $augrandhe --data_dir $datadir

$PY $script --tot test $GPU $N --mp 1 --nccl 1 --gpu_frac $gpufrac --batch_size 500 --model_name res80 --load "$model" --unknown_dir SP-19-2811 --depth $depth --class_weights 1,1 --crop_per_case $croppercase --multi_crop $multicrop --kernels $kernels --expansion $expansion --aug_norm 0 --aug_randhe $augrandhe --data_dir $datadir

#$PY $script --tot test $GPU $N --mp 1 --nccl 1 --gpu_frac $gpufrac --batch_size 500 --model_name res80 --load "$model" --unknown_dir SP-19-5085 --depth $depth --class_weights 1,1 --crop_per_case $croppercase --multi_crop $multicrop --kernels $kernels --expansion $expansion --aug_norm 0 --aug_randhe $augrandhe --data_dir $datadir

$PY $script --tot test $GPU $N --mp 1 --nccl 1 --gpu_frac $gpufrac --batch_size 500 --model_name res80 --load "$model" --unknown_dir SP-19-8085 --depth $depth --class_weights 1,1 --crop_per_case $croppercase --multi_crop $multicrop --kernels $kernels --expansion $expansion --aug_norm 0 --aug_randhe $augrandhe --data_dir $datadir

$PY $script --tot test $GPU $N --mp 1 --nccl 1 --gpu_frac $gpufrac --batch_size 500 --model_name res80 --load "$model" --unknown_dir A --depth $depth --class_weights 1,1 --crop_per_case $croppercase --multi_crop $multicrop --kernels $kernels --expansion $expansion --aug_norm 0 --aug_randhe $augrandhe --data_dir $datadir

#$PY $script --tot test $GPU $N --mp 1 --nccl 1 --gpu_frac $gpufrac --batch_size 500 --model_name res80 --load "$model" --unknown_dir B --depth $depth --class_weights 1,1 --crop_per_case $croppercase --multi_crop $multicrop --kernels $kernels --expansion $expansion --aug_norm 0 --aug_randhe $augrandhe --data_dir $datadir

#$PY $script --tot test $GPU $N --mp 1 --nccl 1 --gpu_frac $gpufrac --batch_size 500 --model_name res80 --load "$model" --unknown_dir C --depth $depth --class_weights 1,1 --crop_per_case $croppercase --multi_crop $multicrop --kernels $kernels --expansion $expansion --aug_norm 0 --aug_randhe $augrandhe --data_dir $datadir

#$PY $script --tot test $GPU $N --mp 1 --nccl 1 --gpu_frac $gpufrac --batch_size 500 --model_name res80 --load "$model" --unknown_dir P --depth $depth --class_weights 1,1 --crop_per_case $croppercase --multi_crop $multicrop --kernels $kernels --expansion $expansion --aug_norm 0 --aug_randhe $augrandhe --data_dir $datadir

#$PY $script --tot test $GPU $N --mp 1 --nccl 1 --gpu_frac $gpufrac --batch_size 500 --model_name res80 --load "$model" --unknown_dir Q --depth $depth --class_weights 1,1 --crop_per_case $croppercase --multi_crop $multicrop --kernels $kernels --expansion $expansion --aug_norm 0 --aug_randhe $augrandhe --data_dir $datadir

#$PY $script --tot test $GPU $N --mp 1 --nccl 1 --gpu_frac $gpufrac --batch_size 500 --model_name res80 --load "$model" --unknown_dir R --depth $depth --class_weights 1,1 --crop_per_case $croppercase --multi_crop $multicrop --kernels $kernels --expansion $expansion --aug_norm 0 --aug_randhe $augrandhe --data_dir $datadir

#$PY $script --tot test $GPU $N --mp 1 --nccl 1 --gpu_frac $gpufrac --batch_size 500 --model_name res80 --load "$model" --unknown_dir SP --depth $depth --class_weights 1,1 --crop_per_case $croppercase --multi_crop $multicrop --kernels $kernels --expansion $expansion --aug_norm $augrandhe

#$PY $script --tot test $GPU $N --mp 1 --nccl 1 --gpu_frac $gpufrac --batch_size 500 --model_name res80 --load "$model" --unknown_dir J1 --depth $depth --class_weights 1,1 --crop_per_case $croppercase --multi_crop $multicrop --kernels $kernels --expansion $expansion --aug_norm 0 --aug_randhe $augrandhe --data_dir $datadir

#$PY $script --tot test $GPU $N --mp 1 --nccl 1 --gpu_frac $gpufrac --batch_size 500 --model_name res80 --load "$model" --unknown_dir J2 --depth $depth --class_weights 1,1 --crop_per_case $croppercase --multi_crop $multicrop --kernels $kernels --expansion $expansion --aug_norm 0 --aug_randhe $augrandhe --data_dir $datadir

#$PY $script --tot test $GPU $N --mp 1 --nccl 1 --gpu_frac $gpufrac --batch_size 500 --model_name res80 --load "$model" --unknown_dir I --depth $depth --class_weights 1,1 --crop_per_case $croppercase --multi_crop $multicrop --kernels $kernels --expansion $expansion --aug_norm 0 --aug_randhe $augrandhe --data_dir $datadir

#$PY $script --tot test $GPU $N --mp 1 --nccl 1 --gpu_frac $gpufrac --batch_size 500 --model_name res80 --load "$model" --unknown_dir L --depth $depth --class_weights 1,1 --crop_per_case $croppercase --multi_crop $multicrop --kernels $kernels --expansion $expansion --aug_norm 0 --aug_randhe $augrandhe --data_dir $datadir

$PY $script --tot test $GPU $N --mp 1 --nccl 1 --gpu_frac $gpufrac --batch_size 500 --model_name res80 --load "$model" --unknown_dir K --depth $depth --class_weights 1,1 --crop_per_case $croppercase --multi_crop $multicrop --kernels $kernels --expansion $expansion --aug_norm 0 --aug_randhe $augrandhe --data_dir $datadir

$PY $script --tot test $GPU $N --mp 1 --nccl 1 --gpu_frac $gpufrac --batch_size 500 --model_name res80 --load "$model" --unknown_dir M --depth $depth --class_weights 1,1 --crop_per_case $croppercase --multi_crop $multicrop --kernels $kernels --expansion $expansion --aug_norm 0 --aug_randhe $augrandhe --data_dir $datadir

#$PY $script --tot test $GPU $N --mp 1 --nccl 1 --gpu_frac $gpufrac --batch_size 500 --model_name res80 --load "$model" --unknown_dir P --depth $depth --class_weights 1,1 --crop_per_case $croppercase --multi_crop $multicrop --kernels $kernels --expansion $expansion --aug_norm 0 --aug_randhe $augrandhe --data_dir $datadir

#$PY $script --tot test $GPU $N --mp 1 --nccl 1 --gpu_frac $gpufrac --batch_size 500 --model_name res80 --load "$model" --unknown_dir R --depth $depth --class_weights 1,1 --crop_per_case $croppercase --multi_crop $multicrop --kernels $kernels --expansion $expansion --aug_norm 0 --aug_randhe $augrandhe --data_dir $datadir
