import torch
from torch import nn
from torch.optim import Adam
import datetime
import sys
import time
import tqdm
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os
from model import ddsm as ddsm
from model.lightning_model_diffusion import LightningDiffusion_augment as lightning_dif
import logging
import lightning as L
import argparse 
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
parser = argparse.ArgumentParser()

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, X, Y1, Y2):
        'Initialization'
        self.X = X
        self.Y1 = Y1
        self.Y2 = Y2

  def __len__(self):
        'Denotes the total number of samples'
        return self.X.shape[0]

  def __getitem__(self, index):
        'Generates one sample of data'
        X = self.X[index,:,:]
        y1 = self.Y1[index]
        y2 = self.Y2[index]
        return (X, y1, y2)


####### Make Lightning Module ########

def list_of_ints(arg):
    return list(map(int, arg.split(',')))

if __name__ == '__main__':
    parser.add_argument('--checkpoint_path', type=str,
                        default='save_models/ATAC/Hepg2_10class_V1/_2024.09.11_03.01.28/diffusion_epoch=479-average-loss=0.285.ckpt')
    parser.add_argument("--save_folder", type=str, default='./save_models')
    parser.add_argument("--train_data_hepg2", type=str, default='tutorials/Human-enhancer/artifacts/DNA-dataset:v0/y_HepG2_10class_atac_clsfree.npz')
    parser.add_argument("--train_data_sknsh", type=str, default='tutorials/Human-enhancer/artifacts/DNA-dataset:v0/y_SKNSH_10class_atac_clsfree.npz')
    parser.add_argument("--device", type=list_of_ints, default=[1])
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--run_name", type=str, default="test" )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--additional_embed_lr", type=float, default=1e-2)

    args = parser.parse_args()

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    run_name = args.run_name
    run_name += f"_{unique_id}"

    wandb.init(project="Diffusion-DNA-cls_free", name=run_name)

    save_folder = args.save_folder + '/' + run_name
    cuda_target = args.device
    max_epochs = args.num_epochs

    class ModelParameters:
        diffusion_weights_file = 'tutorials/Human-enhancer/artifacts/DNA-dataset:v0/steps400.cat4.speed_balance.time4.0.samples100000.pth'  
        time_schedule = 'tutorials/Human-enhancer/artifacts/DNA-dataset:v0/time_dependent.npz'
        device = cuda_target
        batch_size = 256
        num_workers = 4
        n_time_steps = 400
        random_order = False
        speed_balanced = True
        ncat = 4
        lr = args.lr
        additional_embed_lr = args.additional_embed_lr
        checkpoint_path = args.checkpoint_path
    
    config = ModelParameters()

    torch.set_default_dtype(torch.float32)
    
    ###### Pre-parare Dataset ########
    train_ds = np.load(args.train_data_hepg2)['x']
    y1 = np.load(args.train_data_hepg2)['y']
    y2 = np.load(args.train_data_sknsh)['y']
    training_set = Dataset(train_ds, y1, y2)
    data_loader = torch.utils.data.DataLoader(training_set, batch_size = config.batch_size , shuffle=True, num_workers=4)
    
    ################### Start Training ###################
    
    checkpoint = torch.load(config.checkpoint_path, map_location='cpu')
    
    model = lightning_dif(
        weight_file=config.diffusion_weights_file,
        time_schedule=config.time_schedule,
        speed_balanced=config.speed_balanced,
        augment=True,
        ncat=config.ncat, 
        n_time_steps=config.n_time_steps,
        lr=config.lr,
        additional_embed_lr=config.additional_embed_lr,
        continuous=False,
        all_class_number=10,
    )

    # Load the state dictionary
    old_model_dict = model.state_dict()
    for k, v in checkpoint['state_dict'].items():
        if k not in {'model.additional_embed_class.weight'}:
            old_model_dict[k] = v

    model.load_state_dict(old_model_dict)
    
    checkpoint_callback = ModelCheckpoint(
            dirpath=save_folder,
            monitor='average-loss', 
            save_top_k=-1, 
            filename="diffusion_{epoch:03d}-{average-loss:.3f}" 
        )
    
    early_stopping_callback = EarlyStopping(
            monitor='average-loss',  # Metric to monitor
            patience=10,             # Number of epochs to wait for improvement
            verbose=True,
            mode='min',            # Mode can be 'min' or 'max'  
        )
    
    trainer = L.Trainer(accelerator="cuda", 
                        devices=cuda_target, 
                        callbacks=[checkpoint_callback, early_stopping_callback], 
                        max_epochs = max_epochs 
                    )
    trainer.fit(model, data_loader)