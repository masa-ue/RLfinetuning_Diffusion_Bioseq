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
                        default='save_models/ATAC/Hepg2_10class_V1/_2024.09.11_03.01.28/diffusion_epoch=479-average-loss=0.285.ckpt')
    
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument("--guidance_strength", type=float, default=1.0)
    parser.add_argument("--additional_embed_lr", type=float, default=1e-2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--validation_epoch", type=int, default=4)
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--KL_weight", type=float, default=0.0)
    
    parser.add_argument('--save_folder', type=str, default='logs/RL-condition_discrete_ATAC')
    
    return parser.parse_args()

def main():
    args = parse_args()

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    
    run_name = f"gamma={args.guidance_strength}/bs={args.batch_size * args.gradient_accumulation_steps}/lr=({args.lr:g}, {args.additional_embed_lr:g})/_{unique_id}"
    
    save_folder = os.path.join(args.save_folder, unique_id)
    save_folder_eval = os.path.join(save_folder, 'eval_vis')

    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        os.makedirs(save_folder_eval)
    
    wandb.init(entity='sarosavo', project="Finetune-DNA-new-ATAC", \
               name=run_name, config=args)
    
    config = wandb.config

    diffusion_weights_file = 'tutorials/Human-enhancer/artifacts/DNA-dataset:v0/steps400.cat4.speed_balance.time4.0.samples100000.pth'
    time_schedule = 'tutorials/Human-enhancer/artifacts/DNA-dataset:v0/time_dependent.npz'
    
    DEVICE = config.device
    
    reward_model = LightningModel.load_from_checkpoint(
        'tutorials/Human-enhancer/experiment/lightning_logs/4l86okh7/checkpoints/epoch=3-step=1844.ckpt'
    )
    reward_model.eval()
    reward_model.requires_grad_(False)
    reward_model.to(DEVICE)
    
    eval_model = LightningModel.load_from_checkpoint("tutorials/Human-enhancer/artifacts/binary_atac_reward_model.ckpt")
    eval_model.eval()
    eval_model.requires_grad_(False)
    eval_model.to(DEVICE)
    
    # Function to calculate rewards for each condition
    def calculate_rewards(samples_condition):
        rewards = []
        data_loader = torch.utils.data.DataLoader(
            samples_condition.astype("float32"), 
            batch_size=128, 
            num_workers=0
        )
        for batch in data_loader:
            batch = (batch > 0.5) * torch.ones_like(batch)
            batch = torch.permute(batch, (0, 2, 1)).to(DEVICE)
            rewards.append(eval_model(batch).detach().cpu())
        
        return np.concatenate(rewards)

    def new_reward_model(x):
        x = torch.nn.functional.softmax(x / 0.1, -1)  # x: [128, 50, 4]
        seq = torch.transpose(x, 1, 2)  # seq: [128, 4, 50]
        return reward_model(seq)

    original_model = lightning_dif.load_from_checkpoint(
        checkpoint_path=config.checkpoint_path, 
        weight_file=diffusion_weights_file, 
        time_schedule=time_schedule, 
        augment=False,
        continuous=False,
        all_class_number=10,
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
        continuous=False,
        all_class_number=10,
    )

    # Load the state dictionary
    model_dict = new_lightning_dif_model.state_dict()

    # Update the model state dictionary with the checkpoint values, except the additional embed parameters
    for k, v in checkpoint['state_dict'].items():
        if k not in {'model.additional_embed_class.weight'}:
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
    
    assert score_model.additional_embed_class.weight.requires_grad == True
    
    best_loss = 999
    
    # optim = torch.optim.Adam(score_model.parameters(), lr=config.lr)
    optim = torch.optim.Adam([
        {'params': score_model.additional_embed_class.parameters(), 'lr': config.additional_embed_lr},  # Custom learning rate for additional_embed_class
        {'params': [param for name, param in score_model.named_parameters() 
                if "additional_embed_class" not in name], 'lr': config.lr},
    ])

    for k in tqdm(range(config.num_epoch), desc="Epochs"): 
        score_model.train()
        torch.set_grad_enabled(True)
        
        # existing_condition = torch.randint(0,10,(config.batch_size,)).to(DEVICE)  ## Random condition
        existing_condition = torch.randint(0, 2, (config.batch_size,)).to(DEVICE)*9
        
        additional_condition = torch.cat((
            torch.zeros(config.batch_size // 2, dtype=torch.long),
            torch.full((config.batch_size // 2,), 9, dtype=torch.long)
        )).to(DEVICE)
        # additional_condition = torch.full((config.batch_size,), 0, dtype=torch.long).to(DEVICE)
        
        
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
                                    device=DEVICE,
                                    continuous=False
                                )
        
        probabilities = new_reward_model(logits_pred).squeeze(-1)  # (batch, 200, 4) -> (batch, 10)
        
        # choose between two groups of classes or just two classes
        # selected_probs = probabilities[torch.arange(probabilities.size(0)), additional_condition]
        
        selected_probs = torch.zeros(probabilities.size(0)).to(DEVICE)
        group_1 = torch.tensor([0, 1, 2, 3, 4]).to(DEVICE)
        group_2 = torch.tensor([5, 6, 7, 8, 9]).to(DEVICE)
        for i in range(probabilities.size(0)):
            if additional_condition[i] == 0:
                selected_probs[i] = probabilities[i, group_1].sum()  # Sum for classes 0 to 4
            else:
                selected_probs[i] = probabilities[i, group_2].sum()  # Sum for classes 5 to 9
        
        reward = torch.log(selected_probs)
        
        loss = -torch.mean(reward - config.KL_weight * entropy)/config.gradient_accumulation_steps

        loss.backward()

        wandb.log({
            'Train/epoch': k,
            'Train/reward': torch.mean(reward).item(),
            'Train/loss': loss.item()*config.gradient_accumulation_steps,
            'Train/entropy': torch.mean(entropy).item(),
            'Train/embedding_norm': torch.norm(score_model.embed_class.weight).item(),
            'Train/additional_embedding_norm': torch.norm(score_model.additional_embed_class.weight).item(),
        }, step=k)
        
        # Save checkpoint for best loss
        if loss.item() < best_loss:
            best_loss = loss.item()*config.gradient_accumulation_steps
            print(f'Best loss updated: {best_loss:.2f}')
            torch.save(score_model.state_dict(), save_folder + "/best_model.pth")
        
        ### Update (Gradient accumulation)
        if  (k+1) % config.gradient_accumulation_steps == 0:
            optim.step()
            optim.zero_grad()
            
            if (k+1) % config.validation_epoch == 0:
                # Evaluation: create conditions for 10 classes
                conditions_10class = [torch.tensor([i,] * config.batch_size, device=DEVICE, dtype=torch.long) for i in range(10)]
                allsamples_original_conditions_low = [[] for _ in range(10)]
                allsamples_original_conditions_high = [[] for _ in range(10)]
                
                score_model.eval()
                with torch.no_grad():
                    for class_num in range(10):
                        for _ in range(2):
                            samples = ddsm.Euler_Maruyama_sampler(
                                                score_model,
                                                (200, 4), 
                                                existing_condition = conditions_10class[class_num],
                                                additional_condition = torch.cat((
                                                        torch.zeros(config.batch_size // 2, dtype=torch.long),
                                                        torch.full((config.batch_size // 2,), 9, dtype=torch.long)
                                                    )).to(DEVICE),
                                                class_number=10,
                                                strength=config.guidance_strength,
                                                batch_size=config.batch_size,
                                                max_time=4.0,
                                                min_time=1/400,
                                                time_dilation=1,
                                                num_steps=100, 
                                                eps=1e-5,
                                                speed_balanced=True,
                                                device=DEVICE,
                                                augment=True
                                            ).cpu().detach().numpy()
                            allsamples_original_conditions_low[class_num].append(samples[:config.batch_size // 2, :, :])
                            allsamples_original_conditions_high[class_num].append(samples[config.batch_size // 2 :, :, :])
                    
                # Concatenate samples for each class
                allsamples_conditions_low = [np.concatenate(allsamples_original_conditions_low[i], axis=0) for i in range(10)]
                allsamples_conditions_high = [np.concatenate(allsamples_original_conditions_high[i], axis=0) for i in range(10)]
                
                rewards_conditions_low = [calculate_rewards(allsamples_conditions_low[i]) for i in range(10)]
                rewards_conditions_high = [calculate_rewards(allsamples_conditions_high[i]) for i in range(10)]

                # Prepare data for comparison and plotting
                compare_hepg2_low = np.concatenate([rewards_conditions_low[i][:, 1].reshape(-1) for i in range(10)], axis=0)
                compare_hepg2_high = np.concatenate([rewards_conditions_high[i][:, 1].reshape(-1) for i in range(10)], axis=0)
                compare_sknsh_low = np.concatenate([rewards_conditions_low[i][:, 5].reshape(-1) for i in range(10)], axis=0)
                compare_sknsh_high = np.concatenate([rewards_conditions_high[i][:, 5].reshape(-1) for i in range(10)], axis=0)
                
                types = [f'{(i+1)*10}%' for i in range(10)]
                
                types_flat_hepg2_low = [t for i, t in enumerate(types) for _ in range(len(rewards_conditions_low[i][:, 0]))]
                types_flat_hepg2_high = [t for i, t in enumerate(types) for _ in range(len(rewards_conditions_high[i][:, 0]))]
                types_flat_sknsh_low = [t for i, t in enumerate(types) for _ in range(len(rewards_conditions_low[i][:, 1]))]
                types_flat_sknsh_high = [t for i, t in enumerate(types) for _ in range(len(rewards_conditions_high[i][:, 1]))]

                # Data dictionaries for each subplot
                data_dict_hepg2_low = {'Condition (HepG2 quantiles)': types_flat_hepg2_low, 'HepG2': compare_hepg2_low}
                data_dict_hepg2_high = {'Condition (HepG2 quantiles)': types_flat_hepg2_high, 'HepG2': compare_hepg2_high}
                data_dict_sknsh_low = {'Condition (HepG2 quantiles)': types_flat_sknsh_low, 'SKNSH': compare_sknsh_low}
                data_dict_sknsh_high = {'Condition (HepG2 quantiles)': types_flat_sknsh_high, 'SKNSH': compare_sknsh_high}

                # Create DataFrames for each plot
                plot_data_hepg2_low = pd.DataFrame(data_dict_hepg2_low)
                plot_data_hepg2_high = pd.DataFrame(data_dict_hepg2_high)
                plot_data_sknsh_low = pd.DataFrame(data_dict_sknsh_low)
                plot_data_sknsh_high = pd.DataFrame(data_dict_sknsh_high)

                # Create a 2x2 figure for subplots
                fig, axs = plt.subplots(2, 2, figsize=(15, 12))

                sns.boxplot(data=plot_data_hepg2_low, 
                            x='Condition (HepG2 quantiles)', 
                            y='HepG2', 
                            palette="Set2", 
                            showfliers=False, ax=axs[0, 0])
                axs[0, 0].set_title('New condition: low SKNSH', fontsize=14)
                axs[0, 0].set_xlabel('Condition (HepG2 quantiles)')
                axs[0, 0].set_ylabel('HepG2')

                sns.boxplot(data=plot_data_hepg2_high, 
                            x='Condition (HepG2 quantiles)', 
                            y='HepG2', 
                            palette="Set2", 
                            showfliers=False, ax=axs[0, 1])
                axs[0, 1].set_title('New condition: high SKNSH', fontsize=14)
                axs[0, 1].set_xlabel('Condition (HepG2 quantiles)')
                axs[0, 1].set_ylabel('HepG2')

                sns.boxplot(data=plot_data_sknsh_low, 
                            x='Condition (HepG2 quantiles)', 
                            y='SKNSH', 
                            palette="Set1", 
                            showfliers=False, ax=axs[1, 0])
                axs[1, 0].set_title('New condition: low SKNSH', fontsize=14)
                axs[1, 0].set_xlabel('Condition (HepG2 quantiles)')
                axs[1, 0].set_ylabel('SKNSH')

                sns.boxplot(data=plot_data_sknsh_high, 
                            x='Condition (HepG2 quantiles)', 
                            y='SKNSH', 
                            palette="Set1", 
                            showfliers=False, ax=axs[1, 1])
                axs[1, 1].set_title('New condition: high SKNSH', fontsize=14)
                axs[1, 1].set_xlabel('Condition (HepG2 quantiles)')
                axs[1, 1].set_ylabel('SKNSH')
                
                plt.tight_layout()

                # Save plot to a temporary file
                plot_file = os.path.join(save_folder_eval, f'boxplot_epoch_{k}.png')
                plt.savefig(plot_file)
                plt.close()

                # Log plot to wandb
                wandb.log({"eval_vis": wandb.Image(plot_file)}, step=k)
                
                torch.save(score_model.state_dict(), save_folder + "/score-model_ckpt=%d.pth"%k)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
