#!/bin/bash

# Hyperparameter values to iterate over
y_high_values=(1 4)
guidance_strength_values=(5 10)
gradient_accumulation_steps_values=(8 16 24)
lr_values=(3e-4 5e-4 1e-3)
KL_weight_values=(1e-1 10)

# Iterate over each combination of hyperparameter values
for y_high in "${y_high_values[@]}"; do
  for guidance_strength in "${guidance_strength_values[@]}"; do
    for gradient_accumulation_steps in "${gradient_accumulation_steps_values[@]}"; do
      for lr in "${lr_values[@]}"; do
        for KL_weight in "${KL_weight_values[@]}"; do
          echo "Running with y_high=${y_high}, guidance_strength=${guidance_strength}, gradient_accumulation_steps=${gradient_accumulation_steps}, lr=${lr}, KL_weight=${KL_weight}"
          python continuous_finetuning.py \
            --y_low -1 \
            --device 'cuda:1' \
            --y_high "${y_high}" \
            --guidance_strength "${guidance_strength}" \
            --gradient_accumulation_steps "${gradient_accumulation_steps}" \
            --lr "${lr}" \
            --KL_weight "${KL_weight}"
        done
      done
    done
  done
done