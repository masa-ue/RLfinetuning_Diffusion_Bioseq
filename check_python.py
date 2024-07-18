import numpy as np
import pandas as pd
import sys
sys.path.append("../../")
sys.path.append("../../src/model")


import wandb # This is optinal 
from src.utils.sequence import seqs_to_one_hot
wandb.login(host = "https://api.wandb.ai")

run = wandb.init() # Change depending on your proejcts
artifact = run.use_artifact('fderc_diffusion/Diffusion-DNA-RNA/DNA-dataset:v0')
dir = artifact.download()
wandb.finish()