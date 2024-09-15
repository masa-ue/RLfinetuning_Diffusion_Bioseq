

# Adding Conditional Control to Diffusion Models with Reinforcement Learning for Biological Sequences  

## Installation

First, install gReLU package (https://github.com/Genentech/gReLU).

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install tqdm pandas matplotlib lightning
```

## Checkpoints and files

Please download the following checkpoints and files and put them in the corresponding folders from [https://drive.google.com/drive/folders/1n9Hknzg45vtzstIqZsVRuWV2-D8V61Zn?usp=sharing](https://drive.google.com/drive/folders/1n9Hknzg45vtzstIqZsVRuWV2-D8V61Zn?usp=sharing).

1. `save_models/ATAC/Hepg2_10class_V1/_2024.09.11_03.01.28/diffusion_epoch=479-average-loss=0.285.ckpt`

2. `tutorials/Human-enhancer/artifacts/binary_atac_reward_model.ckpt`

3. `tutorials/Human-enhancer/experiment/lightning_logs/4l86okh7/checkpoints/epoch=3-step=1844.ckpt`

4. `tutorials/Human-enhancer/artifacts/DNA-dataset:v0/dataset.csv.gz`

5. `tutorials/Human-enhancer/artifacts/DNA-dataset:v0/steps400.cat4.speed_balance.time4.0.samples100000.pth`

6. `tutorials/Human-enhancer/artifacts/DNA-dataset:v0/time_dependent.npz`

## What is included?

Our goal: to condition generations on a new condition ($y$), given a pre-trained diffusion model on an exisiting condition (i.e. $p(x|c)$) and a classifier of the new condition ($x \mapsto y$).  
The ultimate goal is to sample from $p(x|c,y)$. We are in a fine-tuning setting, where we have limited data labeled with the new condition.

### 1. Pre-trained conditional diffusion model on HepG2

A pre-trained diffusion model on HepG2 is provided at: `save_models/ATAC/Hepg2_10class_V1/_2024.09.11_03.01.28/diffusion_epoch=479-average-loss=0.285.ckpt`

Training of this model is needs to relabel sequence data with the Binary-ATAC-reward_model, which can be found at: `tutorials/Human-enhancer/artifacts/binary_atac_reward_model.ckpt`.
For more details on how to use this reward model, you can refer to `tutorials/Human-enhancer/1-Enhancer_data_relabel-atac.ipynb`.

### 2. New classifier on a new condition: SKNSH

The notebook `tutorials/Human-enhancer/2-Train_classifier.ipynb` shows how to train a classifier on a new condition (SKNSH) using the data relabeled with the Binary-ATAC-model. (Note: running this notebook requires to fix some bugs of the gReLU package related to multiclass classification. Consider installing from a patched version [https://github.com/zhaoyl18/gReLU]).

For convenience, a classifier checkpoint is provided at: `tutorials/Human-enhancer/experiment/lightning_logs/4l86okh7/checkpoints/epoch=3-step=1844.ckpt`.  
You may refer to `tutorials/Human-enhancer/1-prepare_data-clsfree-new.ipynb` for how to use this classifier. In this notebook, we show how to use the classifier to predict the class of a new condition (SKNSH) for sequences generated from the pre-trained diffusion model.

### 3. Our proposal: CTRL

Completed, see `discrete_finetuning_atac.py`.

### 4. Baselines

#### 4.1. Classifier-free guidance

Completed.

1. `tutorials/Human-enhancer/1-prepare_data-clsfree-new.ipynb` shows how to augment data with the pre-trained diffusion model and the classifier (i.e., $(c,x,y)$).

2. `src/lightning_diffusion_clsfree.py` and `scripts/conditionalDM_training.sh` show how to perform classifier-free finetuning with the augmented data.  

3. `tutorials/Human-enhancer/Evaluate_clsfree-baseline.ipynb` shows how to evaluate this baseline by plotting the boxplots across (1) 10 classes of the old condition (HepG2) and (2) class $0$ and class $9$ of the new condition (SKNSH).  

#### 4.2. TODO: DPS and/or classifier-based guidance  

1. First, this method will use the pre-trained diffusion model (on HepG2) and the classifier (on SKNSH).  

2. Second, this is an inference-time technique. This method will inject the gradient information from the classifier to to the denoising process of the diffusion model.  

3. Therefore, it is recommended to write a new sampler or consider overwrite the diffusion model's forward method to add additional gradient to the existing score function.  

4. To start, a useful template of how to load pre-trained model and classifier from checkpoints will be `tutorials/Human-enhancer/1-prepare_data-clsfree-new.ipynb`. Probably can work with the ***Euler_Maruyama_sampler***, see `src/model/ddsm.py`.

5. The evaluation of this method will be similar to the classifier-free guidance. Consider evaluating by following `tutorials/Human-enhancer/Evaluate_clsfree-baseline.ipynb`.  

