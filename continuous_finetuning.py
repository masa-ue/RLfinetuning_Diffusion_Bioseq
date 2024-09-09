import torch
import numpy as np
import pandas as pd
import datetime
import sys
import os
import argparse
from scipy.stats import pearsonr

from matplotlib import pyplot as plt

import seaborn as sns
sns.set(style="whitegrid")

from src.model import ddsm as ddsm
from src.model.ddsm_fine_tune import Euler_Maruyama_sampler_GPU_Conditional

from src.model import ddsm_model as modeld
from src.model.lightning_model_diffusion import LightningDiffusion as lightning_dif
import src.utils.sequence as utils

from grelu.lightning import LightningModel

import wandb
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str,
                        default='./save_models/hepg2_continuous/V2/-1_4-lr=1e-3/_2024.07.21_15.05.16/diffusion_epoch=156-average-loss=0.30.ckpt')
    
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--y_low', type=float, default=-1.0)
    parser.add_argument('--y_high', type=float, default=4.0)
    
    parser.add_argument("--guidance_strength", type=float, default=5.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_epoch", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--KL_weight", type=float, default=0.0)
    
    parser.add_argument('--save_folder', type=str, default='logs/RL-condition_continuous_v3')
    
    return parser.parse_args()

def main():
    args = parse_args()

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    
    run_name = f"gamma={args.guidance_strength}_Y={(args.y_low,args.y_high)}_bs={args.batch_size * args.gradient_accumulation_steps}_lr={args.lr:g}_KL={args.KL_weight}_{unique_id}"
    
    save_folder = os.path.join(args.save_folder, unique_id)
    save_folder_eval = os.path.join(save_folder, 'eval_vis')

    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        os.makedirs(save_folder_eval)
    
    wandb.login(host="https://genentech.wandb.io")
    wandb.init(entity='zhao-yulai', project="Finetune-DNA-v3", \
               name=run_name, config=args)
    
    config = wandb.config

    diffusion_weights_file = 'tutorials/Human-enhancer/artifacts/MPRA-dataset:v2/steps400.cat4.speed_balance.time4.0.samples100000.pth'
    time_schedule = 'tutorials/Human-enhancer/artifacts/MPRA-dataset:v2/time_dependent.npz'
    
    DEVICE = config.device
    
    reward_model = LightningModel.load_from_checkpoint(
        "./save_models/eval_reward_models/hepg2_regressor.ckpt"
    )

    reward_model.eval()
    reward_model.requires_grad_(False)
    reward_model.to(DEVICE)

    def new_reward_model(x):
        x = torch.nn.functional.softmax(x / 0.1, -1)  # x: [128, 50, 4]
        seq = torch.transpose(x, 1, 2)  # seq: [128, 4, 50]
        return reward_model(seq)

    original_model = lightning_dif.load_from_checkpoint(
        checkpoint_path=config.checkpoint_path, 
        weight_file=diffusion_weights_file, 
        time_schedule=time_schedule, 
        augment=False,
        continuous=True,
        y_low=-1.0,
        y_high=4.0
    )

    original_model = original_model.model
    original_model.cuda(device=DEVICE)
    
    # Freeze all parameters of the original model
    original_model.eval()
    for param in original_model.parameters():
        param.requires_grad = False

    # Load the checkpoint
    checkpoint = torch.load(config.checkpoint_path, map_location='cpu')

    # Create the model instance
    new_lightning_dif_model = lightning_dif(
        weight_file=diffusion_weights_file,
        time_schedule=time_schedule,
        augment=True,
        continuous=True,
        y_low=-1.0,
        y_high=4.0
    )

    # Load the state dictionary
    model_dict = new_lightning_dif_model.state_dict()

    # Update the model state dictionary with the checkpoint values, except the additional embed parameters
    for k, v in checkpoint['state_dict'].items():
        if k not in {'model.additional_embed.0.W', 'model.additional_embed.1.bias', 'model.additional_embed.1.weight'}:
            model_dict[k] = v

    new_lightning_dif_model.load_state_dict(model_dict)

    score_model = new_lightning_dif_model.model
    score_model.cuda(device=DEVICE)  # Move the model to the GPU

    # freeze old condition's embeddings
    score_model.embed_class.requires_grad_(False)
    score_model.embed_class.eval()
    
    # freeze time's embeddings
    score_model.embed.requires_grad_(False)
    score_model.embed.eval()

    assert score_model.additional_embed[1].weight.requires_grad == True

    # starting fine-tuning
    eval_pearson_old = []
    eval_pearson_new = []
    
    best_loss = 999
    optim = torch.optim.Adam(score_model.parameters(), lr=config.lr)

    for k in tqdm(range(config.num_epoch), desc="Epochs"): 
        score_model.train()
        torch.set_grad_enabled(True)
        
        existing_condition = torch.empty(config.batch_size, device=DEVICE).uniform_(-1, args.y_high)
        additional_condition = torch.empty(config.batch_size, device=DEVICE).uniform_(-1, args.y_high)
        
        # Training 
        logits_pred, entropy = Euler_Maruyama_sampler_GPU_Conditional(
                                    score_model, 
                                    original_model,
                                    (200, 4),
                                    existing_condition=existing_condition,
                                    additional_condition=additional_condition,
                                    strength=config.guidance_strength,
                                    batch_size=config.batch_size,
                                    max_time=4.0,
                                    min_time=1/400,
                                    time_dilation=1,
                                    num_steps=50, 
                                    gradient_start=45,
                                    eps=1e-5,
                                    speed_balanced=True,
                                    device=DEVICE
                                )
        
        raw_reward = new_reward_model(logits_pred)[:,1].squeeze(1) #### logits_pred: [128, 50, 4] -> reward: [128,3,1]
        cls_reward = -torch.square(raw_reward - additional_condition)
        
        loss = -torch.mean(cls_reward - config.KL_weight * entropy)/config.gradient_accumulation_steps
        entropy_accum = torch.mean(entropy)/config.gradient_accumulation_steps
        
        pearson_k562, _ = pearsonr(raw_reward.detach().cpu().numpy(), 
                        additional_condition.cpu().numpy())
        
        hepg2_rewards = new_reward_model(logits_pred)[:,0].squeeze(1)
        pearson_hepg2, _ = pearsonr(hepg2_rewards.detach().cpu().numpy(), 
                        existing_condition.cpu().numpy())
        
        # print('Epoch: %d | Loss: %.2f | entropy: %.2f | K562 Pearson %.2f \n'%(k, loss.item(), entropy_accum.item(), pearson_k562))

        wandb.log({
            'Train/epoch': k,
            'Train/loss': loss.item(),
            'Train/entropy': entropy_accum.item(),
            'Train/K562 Pearson': pearson_k562,
            'Train/HepG2 Pearson': pearson_hepg2
        })

        loss.backward()
        
        # Save checkpoint for best loss
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_pearson_old = pearson_hepg2
            best_pearson_new = pearson_k562
            
            print(f'Best loss updated: {best_loss:.2f},\
                  HepG2 Pearson: {best_pearson_old:.2f}, \
                  K562 Pearson: {best_pearson_new:.2f}')
            
            torch.save(score_model.state_dict(), save_folder + "/model_ckpt=%d.pth"%k)
        
        ### Update (Gradient accumulation)
        if  (k+1) % config.gradient_accumulation_steps == 0:
            optim.step()
            optim.zero_grad()
            
            # Evaluation
            total_samples = 2 * config.batch_size
                    
            eval_existing_condition = torch.empty(total_samples, device=DEVICE).uniform_(-1, args.y_high)
            eval_additional_condition = torch.empty(total_samples, device=DEVICE).uniform_(-1, args.y_high)
            
            with torch.no_grad():
                score_model.eval()
                logits_pred  = ddsm.Euler_Maruyama_sampler(
                                    score_model,
                                    (200, 4), 
                                    existing_condition = eval_existing_condition,
                                    additional_condition = eval_additional_condition,
                                    class_number=0,
                                    strength=config.guidance_strength,
                                    batch_size=total_samples,
                                    max_time=4.0,
                                    min_time=1/400,
                                    time_dilation=1,
                                    num_steps=50, 
                                    eps=1e-5,
                                    speed_balanced=True,
                                    device=DEVICE
                                )
                
                preds_onehot = (logits_pred > 0.5) * torch.ones_like(logits_pred)
                preds_onehot = torch.permute(preds_onehot, (0, 2, 1)).to(DEVICE)
                
                raw_reward_old_condition = reward_model(preds_onehot)[:,0].squeeze(1)
                raw_reward_new_condition = reward_model(preds_onehot)[:,1].squeeze(1)

            pearson_old, _ = pearsonr(raw_reward_old_condition.detach().cpu().numpy(), eval_existing_condition.cpu().numpy())
            pearson_new, _ = pearsonr(raw_reward_new_condition.detach().cpu().numpy(), eval_additional_condition.cpu().numpy())
            
            print('Evaluation ------ HepG2 Pearson: %.3f | K562 Pearson: %.3f\n'%(pearson_old, pearson_new))
            
            eval_pearson_old.append(pearson_old)
            eval_pearson_new.append(pearson_new)

            wandb.log({
                'Eval/HepG2_Pearson': pearson_old,
                'Eval/K562_Pearson': pearson_new
            })
            
            # Plotting and logging scatter plots
            plt.figure(figsize=(12, 6))
            plt.suptitle(f'Evaluation (Epoch: {k})', fontsize=28)

            # First subplot
            ax1 = plt.subplot(1, 2, 1)
            sns.scatterplot(
                x=eval_existing_condition.detach().cpu().numpy(), 
                y=raw_reward_old_condition.detach().cpu().numpy(), 
                alpha=0.8,
                label=f'Pearson = {pearson_old:.2f}'
            )
            ax1.set_xlabel('Condition', fontsize=22)
            ax1.set_ylabel('Score', fontsize=22)
            ax1.set_title('HepG2', fontsize=26)
            ax1.grid(True, linestyle='--', linewidth=1)
            ax1.tick_params(axis='both', which='major', labelsize=18)
            ax1.legend(loc='upper left', fontsize=20)

            # Second subplot
            ax2 = plt.subplot(1, 2, 2)
            sns.scatterplot(
                x=eval_additional_condition.detach().cpu().numpy(), 
                y=raw_reward_new_condition.detach().cpu().numpy(), 
                alpha=0.8,
                label=f'Pearson = {pearson_new:.2f}'
            )
            ax2.set_xlabel('Condition', fontsize=22)
            ax2.set_ylabel('Score', fontsize=22)
            ax2.set_title('K562', fontsize=26)
            ax2.grid(True, linestyle='--', linewidth=1)
            ax2.tick_params(axis='both', which='major', labelsize=18)
            ax2.legend(loc='upper left', fontsize=20)
            
            plt.tight_layout(pad=2.0)
            # plt.style.use('dark_background')

            # Save plot to a temporary file
            plot_file = os.path.join(save_folder_eval, f'scatter_plot_epoch_{k}.png')
            plt.savefig(plot_file)
            plt.close()

            # Log plot to wandb
            wandb.log({"eval_vis": wandb.Image(plot_file)})

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()