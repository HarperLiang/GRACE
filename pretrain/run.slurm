#!/bin/bash

#SBATCH --job-name=dino_lora  
#SBATCH -o /path/to/job-%j.log
#SBATCH -e /path/to/job-%j.err
#SBATCH --time=168:00:00
#SBATCH -N 1 -n 64 --gres=gpu:8
#SBATCH --mem-per-cpu=4G

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $CONDA_PREFIX/bin/torchrun --nproc_per_node=8 main_dino.py \
                                              --arch virchow2 \
                                              --data_path /path/to/data \
                                              --output_dir /path/to/output \
                                              --saveckp_freq 1 \
                                              --batch_size_per_gpu 48 \
                                              --patch_size 14 \
                                              --local_crops_size 98 > /path/to/log.log 2>&1 
