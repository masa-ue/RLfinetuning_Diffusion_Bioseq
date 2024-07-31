python continuous_finetuning.py \
            --y_low -1 \
            --device 'cuda:4' \
            --y_high 4 \
            --guidance_strength 10 \
            --gradient_accumulation_steps 4 \
            --lr 3e-4 \
            --KL_weight 1e-2 \
            --num_epoch 2000

python continuous_finetuning.py \
            --y_low -1 \
            --device 'cuda:1' \
            --y_high 4 \
            --guidance_strength 5 \
            --gradient_accumulation_steps 8 \
            --lr 3e-4 \
            --KL_weight 1e-2 \
            --num_epoch 2000