#!/bin/bash
python ./src/lightning_diffusion.py \
     --save_folder ./save_models \
     --class_number 0 \
     --device 1 \
     --num_epochs 400 \
     --train_data ./data/y_HepG2_relabel.npz \
     --continuous True \
     --y_min -1 \
     --y_max 4 \
     --lr 3e-3 \
     --run_name hepg2_continuous/V2/-1_4-lr=3e-3/