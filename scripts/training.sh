#!/bin/bash
python ./src/lightning_diffusion.py \
     --save_folder ./save_models \
     --class_number 3 \
     --device 1 \
     --train_data ./data/y_HepG2_3class.npz \
     --run_name V1-lightning/hepg2_3class_earlystop