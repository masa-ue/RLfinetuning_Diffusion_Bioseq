import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset, Dataset
from torch.optim import Adam

import time
import tqdm
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

import sys
sys.path.append("../")
import model.ddsm as ddsm
import model.ddsm_model as modeld
import logging

logger = logging.getLogger()

import argparse 
parser = argparse.ArgumentParser()


parser.add_argument("--save_folder", type=str, help="Folder to be saved")
parser.add_argument("--train_data", type = str)
parser.add_argument("--class_number", type = int)
parser.add_argument("--device", type = str, default = "cuda")
parser.add_argument("--wandb", type = str, default = False )
parser.add_argument("--num_epochs", type = int, default = 100 )

args = parser.parse_args()
save_folder = args.save_folder
cuda_target = args.device
all_class_number = args.class_number 
wandb_true = args.wandb
train_folder = args.train_data
num_epochs = args.num_epochs

if wandb_true == "True":  
    import wandb
    wandb.init(entity ='grelu', project="Diffusion-DNA-RNA", name = "test")
    
class ModelParameters:
    diffusion_weights_file = '../data/steps400.cat4.speed_balance.time4.0.samples100000.pth'
    device = cuda_target
    batch_size = 128
    num_workers = 4
    n_time_steps = 400
    random_order = False
    speed_balanced = True
    ncat = 4
    lr = 5e-4
    
config = ModelParameters()

sb = ddsm.UnitStickBreakingTransform()

### LOAD WEIGHTS
v_one, v_zero, v_one_loggrad, v_zero_loggrad, timepoints = torch.load(config.diffusion_weights_file)
v_one = v_one.cpu()
v_zero = v_zero.cpu()
v_one_loggrad = v_one_loggrad.cpu()
v_zero_loggrad = v_zero_loggrad.cpu()
timepoints = timepoints.cpu()
alpha = torch.ones(config.ncat - 1).float()
beta =  torch.arange(config.ncat - 1, 0, -1).float()

time_dependent_weights = torch.tensor(np.load('../data/time_dependent.npz')['x']).to(config.device)

#### TRAINING CODE
score_model = nn.DataParallel(modeld.ScoreNet(time_dependent_weights=torch.sqrt(time_dependent_weights)), device_ids = [config.device])
score_model = score_model.to(config.device)
score_model.train()

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

training_set = Dataset(train_ds, y1 )
data_loader = torch.utils.data.DataLoader(training_set, batch_size = config.batch_size , shuffle=True, num_workers=0)

sampler = ddsm.Euler_Maruyama_sampler

optimizer = Adam(score_model.parameters(), lr=config.lr)

torch.set_default_dtype(torch.float32)
bestsei_validloss = float('Inf')

tqdm_epoch = tqdm.trange(num_epochs)


for epoch in tqdm_epoch:
    avg_loss = 0
    num_items = 0
    stime = time.time()
    for xS, yS in tqdm.tqdm((data_loader)):
        x = xS[:, :, :4]
        # Optional : there are several options for importance sampling here. it needs to match the loss function
        random_t = torch.LongTensor(np.random.choice(np.arange(config.n_time_steps), size=x.shape[0],
                                                     p=(torch.sqrt(time_dependent_weights) / torch.sqrt(
                                                         time_dependent_weights).sum()).cpu().detach().numpy()))
       
        perturbed_x, perturbed_x_grad = ddsm.diffusion_fast_flatdirichlet(x.cpu(), random_t, v_one, v_one_loggrad)
        # perturbed_x, perturbed_x_grad = diffusion_factory(x, random_t, v_one, v_zero, v_one_loggrad, v_zero_loggrad, alpha, beta)

        perturbed_x = perturbed_x.to(config.device)
        perturbed_x_grad = perturbed_x_grad.to(config.device)
        random_timepoints = timepoints[random_t].to(config.device)
        
        yS = yS.type(torch.LongTensor)
        random_list = np.random.binomial(1,0.3, yS.shape[0])
        yS[random_list ==1 ] = all_class_number
        yS = yS.to(config.device)
     
        score = score_model(perturbed_x, random_timepoints, yS)

        # the loss weighting function may change, there are a few options that we will experiment on
        if config.speed_balanced:
            s = 2 / (torch.ones(config.ncat - 1, device=config.device) + torch.arange(config.ncat - 1, 0, -1,
                                                                                      device=config.device).float())
        else:
            s = torch.ones(config.ncat - 1, device=config.device)

        
        perturbed_v = sb._inverse(perturbed_x, prevent_nan=True).detach()
        loss = torch.mean(torch.mean(
            1 / (torch.sqrt(time_dependent_weights))[random_t][(...,) + (None,) * (x.ndim - 1)] * s[
                (None,) * (x.ndim - 1)] * perturbed_v * (1 - perturbed_v) * (
                        ddsm.gx_to_gv(score, perturbed_x, create_graph=True) - ddsm.gx_to_gv(perturbed_x_grad,
                                                                                    perturbed_x)) ** 2, dim=(1)))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]

        if wandb_true == "True": 
            wandb.log({"loss": loss})

        torch.save(score_model.state_dict(), save_folder+"_%d.pth"%epoch )

    # Print the averaged training loss so far.
    print(avg_loss / num_items)
    if wandb_true == "True":   
        wandb.log({"Epoch average loss": avg_loss / num_items})
        wandb.log({"Epoch": epoch} )
    tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))

if wandb_true == "True":
    wandb.finish()