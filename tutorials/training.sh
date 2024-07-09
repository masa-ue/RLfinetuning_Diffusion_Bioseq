#!/bin/bash
python ../src/lightning_diffusion.py \
     --save_folder ../save_models/diffusion/diffusion  --class_number 3 --device 6 \
     --wandb_projectname Diffusion-RNA --num_epochs 100 --wandb True --train_data 'artifacts/RNA-MPRA-dataset:v0/RNA_seq.npz'