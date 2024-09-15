#!/bin/bash
# python ./src/lightning_diffusion.py \
#      --save_folder ./save_models \
#      --class_number 10 \
#      --device 4 \
#      --num_epochs 600 \
#      --train_data tutorials/Human-enhancer/artifacts/DNA-dataset:v0/y_HepG2_10class_atac.npz \
#      --lr 3e-3 \
#      --run_name ATAC/Hepg2_10class_V1/
#      # --continuous \


python ./src/lightning_diffusion_clsfree.py \
     --save_folder ./save_models \
     --device 8 \
     --num_epochs 40 \
     --lr 1e-4 \
     --additional_embed_lr 1e-2 \
     --run_name cls-free/V1/