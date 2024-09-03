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
from model.lightning_model_diffusion import LightningDiffusion as lightning_dif
import logging
import lightning as L
import argparse 
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
parser = argparse.ArgumentParser()

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, X, Y):
        'Initialization'
        self.X = X
        self.Y = Y 

  def __len__(self):
        'Denotes the total number of samples'
        return self.X.shape[0]

  def __getitem__(self, index):
        'Generates one sample of data'
        X = self.X[index,:,:]
        y = self.Y[index]
        return (X, y)


####### Make Lightning Module ########

def list_of_ints(arg):
    return list(map(int, arg.split(',')))

if __name__ == '__main__':
    parser.add_argument("--save_folder", type=str, default='./save_models')
    parser.add_argument("--train_data", type=str, default='./data/y_HepG2_relabel.npz')
    parser.add_argument("--class_number", type=int, default=0)
    parser.add_argument("--device", type=list_of_ints, default=[1])
    parser.add_argument("--num_epochs", type=int, default=100 )
    parser.add_argument("--run_name", type=str, default="test" )
    parser.add_argument("--continuous", action='store_true', default=False)
    parser.add_argument("--y_min", type=float, default=-4.0)
    parser.add_argument("--y_max", type=float, default=10.0)
    parser.add_argument("--lr", type=float, default=5e-4)

    args = parser.parse_args()

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    run_name = args.run_name
    run_name += f"_{unique_id}"

    wandb.init(project="Diffusion-DNA-RNA-new", name=run_name)

    save_folder = args.save_folder + '/' + run_name
    cuda_target = args.device
    all_class_number = args.class_number 
    train_folder = args.train_data
    max_epochs = args.num_epochs
    continuous = args.continuous
    y_min = args.y_min
    y_max = args.y_max
    lr = args.lr

    seed_file_name= 'tutorials/Human-enhancer/artifacts/DNA-dataset:v0/steps400.cat4.speed_balance.time4.0.samples100000.pth'
    time_schedule = 'tutorials/Human-enhancer/artifacts/DNA-dataset:v0/time_dependent.npz'

    class ModelParameters:
        diffusion_weights_file = seed_file_name  
        device = cuda_target
        batch_size = 256
        num_workers = 4
        n_time_steps = 400
        random_order = False
        speed_balanced = True
        ncat = 4
        lr = lr
    
    config = ModelParameters()

    torch.set_default_dtype(torch.float32)
    
    ###### Pre-parare Dataset ########
    train_ds = np.load(train_folder)['x']
    y1 = np.load(train_folder)['y']
    training_set = Dataset(train_ds, y1)
    data_loader = torch.utils.data.DataLoader(training_set, batch_size = config.batch_size , shuffle=True, num_workers=4)
    
    ################### Start Training ###################
    model = lightning_dif(
                        weight_file=config.diffusion_weights_file, 
                        time_schedule=time_schedule, 
                        speed_balanced=config.speed_balanced, 
                        all_class_number=all_class_number,
                        augment=False, 
                        ncat=config.ncat, 
                        n_time_steps=config.n_time_steps, 
                        lr=config.lr,
                        continuous=continuous,
                        y_low=y_min,
                        y_high=y_max
                    )
    
    checkpoint_callback = ModelCheckpoint(
            dirpath=save_folder,
            monitor='average-loss', 
            save_top_k=-1, 
            filename="diffusion_{epoch:03d}-{average-loss:.2f}" 
        )
    
    early_stopping_callback = EarlyStopping(
            monitor='average-loss',  # Metric to monitor
            patience=10,             # Number of epochs to wait for improvement
            verbose=True,
            mode='min'               # Mode can be 'min' or 'max'
        )
    
    trainer = L.Trainer(accelerator="cuda", 
                        devices=cuda_target, 
                        callbacks=[checkpoint_callback, early_stopping_callback], 
                        max_epochs = max_epochs 
                    )
    trainer.fit(model, data_loader)