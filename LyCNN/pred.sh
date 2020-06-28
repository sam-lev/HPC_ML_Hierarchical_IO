#!/bin/bash

#SBATCH --time=21-00:00:00 # walltime, abbreviated by -t                                         
#SBATCH --mem=10G             
#SBATCH -o sclassify-unknown.out-%j # name of the stdout, using the job number (%j) and the first node (%N)             
#SBATCH -e sclassify-unknown.err-%j # name of the stderr, using the job number (%j) and the first node (%N)                     
#SBATCH --gres=gpu:1

model = "$1"


#source ~/.bashrc
#conda activate tf
#export PATH="/home/sci/samlev/Convolutional_Neural_Net_Lymphoma/code:$PATH"

/home/sci/samlev/bin/bin/python3.5 multiThreadDistNNet.py --tot test --gpu 0,1 --batch_size 500 --model_name ep_134_unkown_A --load "$model" --unknown_dir $2 --depth $3 --class_weights 1,1 --multi_crop $4 --crop_per_case 10000

#echo -e "\n--------------------------------------------------------\n" >> "/home/sci/samlev/Convolutional_Neural_Net_Lymphoma/data/Predictions/$2/predictions.txt"

#echo -e "     4 crops from min(10,number imgs in case)     \n" >> "/home/sci/samlev/Convolutional_Neural_Net_Lymphoma/data/Predictions/$2/predictions.txt"

#echo -e "\n--------------------------------------------------------\n" >> "/home/sci/samlev/Convolutional_Neural_Net_Lymphoma/data/Predictions/$2/predictions.txt"

source ~/.bashrc

emacs "/home/sci/samlev/Convolutional_Neural_Net_Lymphoma/data/Predictions/$2/predictions$2.txt"
