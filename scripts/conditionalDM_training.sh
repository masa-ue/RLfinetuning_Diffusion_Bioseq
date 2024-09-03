#!/bin/bash
python ./src/lightning_diffusion.py \
     --save_folder ./save_models \
     --class_number 3 \
     --device 4 \
     --num_epochs 400 \
     --train_data tutorials/Human-enhancer/artifacts/DNA-dataset:v0/y_HepG2_3class.npz \
     --lr 3e-3 \
     --run_name ATAC_Hepg2_3class_V0/
     # --continuous \


# python ./src/lightning_diffusion.py \
#      --save_folder ./save_models \
#      --class_number 0 \
#      --device 1 \
#      --num_epochs 400 \
#      --train_data ./data/y_HepG2_relabel.npz \
#      --continuous \
#      --y_min -1 \
#      --y_max 4 \
#      --lr 3e-3 \
#      --run_name hepg2_continuous/V2/-1_4-lr=3e-3/