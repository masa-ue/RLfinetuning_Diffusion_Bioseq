import torch
from torch import nn
from torch.optim import Adam
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
from lightning.pytorch.callbacks import ModelCheckpoint


# Take Arguments 
parser = argparse.ArgumentParser()

def list_of_ints(arg):
    return list(map(int, arg.split(',')))
 

parser.add_argument("--train_data", type = str)
parser.add_argument("--wandb_projectname", type = str)
parser.add_argument("--class_number", type = int)
parser.add_argument("--device", type = list_of_ints, default = [2])
parser.add_argument("--wandb", type = str, default = True )
parser.add_argument("--num_epochs", type = int, default = 100 )
parser.add_argument("--seed_file_name", type =str )
parser.add_argument("--time_schedule", type= str)

args = parser.parse_args()
cuda_target = args.device
all_class_number = args.class_number 
wandb_true = args.wandb
train_folder = args.train_data
max_epochs = args.num_epochs
wandb_projectname = args.wandb_projectname
seed_file_name= args.seed_file_name
time_schedule = args.time_schedule

class ModelParameters:
    diffusion_weights_file = seed_file_name  
    device = cuda_target
    batch_size = 256
    num_workers = 4
    n_time_steps = 400
    random_order = False
    speed_balanced = True
    ncat = 4
    lr = 5e-4
    
config = ModelParameters()

torch.set_default_dtype(torch.float32)
###### Pre-parare Dataset 

train_ds = np.load(train_folder)['x']
y1 = np.load(train_folder)['y']

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



if __name__ == '__main__':
    #################
    training_set = Dataset(train_ds, y1 )
    data_loader = torch.utils.data.DataLoader(training_set, batch_size = config.batch_size , shuffle=True, num_workers= 4)


    wandb.init(project= wandb_projectname, name = "diffusion") 
        
    ################### Start Training ##############

    model = lightning_dif(config.diffusion_weights_file, time_schedule, config.speed_balanced, all_class_number, config.ncat, config.n_time_steps, config.lr )
    checkpoint_callback = ModelCheckpoint(monitor='average-loss', save_top_k =  -1, filename = "diffusion_{epoch:03d}" )
    trainer = L.Trainer(accelerator= "cuda", devices = cuda_target, callbacks=[checkpoint_callback], max_epochs = max_epochs )
    trainer.fit(model, data_loader )

