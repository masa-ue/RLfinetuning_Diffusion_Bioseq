#!/bin/bash
python ../src/train_conditional.py \
     --save_folder ../save_models/diffusion/diffusion  --class_number 2 --device cuda:7 \
     --wandb True --train_data ../data/y_HepG2.npz