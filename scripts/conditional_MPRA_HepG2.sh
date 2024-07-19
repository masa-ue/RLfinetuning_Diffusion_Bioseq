#!/bin/bash
python ./src/train_conditional.py \
     --save_folder ./save_models  \
     --class_number 3 \
     --device cuda:4 \
     --train_data ./data/y_HepG2_3class.npz \
     --run_name V0-hepg2_3class

# ! python ../../src/train_conditional.py \
#      --save_folder ../save_models/diffusion/diffusion  \
#      --class_number 3 \ 
#      --device cuda:4 \
#      --train_data 'artifacts/MPRA-dataset:v2/y_HepG2_3class.npz'