# RL-Based Fine-Tuning for Bilological Sequences 


This is a code for the tutorial/review paper for RL-based-fine-tuniing.  In this code, we especially focus on the design of biological sequences like DNA (enhancers) and RNA (UTRs) design. 

![Summary](./media/summary.png)

## Tutorials

See notebooks in the tutorial folders. Each notebook is self-contained. There is also the corresponding notebooks for the optimization of enhancers. 

* [1-UTR_data.ipynb](tutorials/UTR/1-UTR_data.ipynb) : Get raw data (x, y) and how we make labels from y
* [2-UTR_diffusion_training.ipynb](tutorials/UTR/2-UTR_diffusion_training.ipynb): Train conditional and unconditional diffuison models (score-based Diffusion over Simplex in )
* [3-UTR_evaluation](tutorials/UTR/3-UTR_evaluation.ipynb): Evaluate the perfomance of conditional generative models
* [4-UTR_finetune_directbackprop](tutorials/UTR/4-UTR_finetune_directbackprop.ipynb): Main fine-tuning code with direct reward backpropagation 
* [5-UTR_finetune_PPO.ipynb](tutorials//UTR/5-UTR_finetune_PPO.ipynb):  Main fine-tuning code with PPO  
* [Oracle_training](tutorials/UTR/UTR_oracle_training.ipynb): Make rewards models from the dataset (sequence, activity)


The following is a UTR sequence before/after fine-tuning. We optimize an MRL (activity level). 

<img src= "./media/RNA_output_high_finetune.png"  width="200"> <img src= "./media/chat_UTR.png" width="300"> 


 The following is a DNA sequence (enhancer) before/after RL-based fine-tuning. We optimize an activity level. 

<img src= "./media/DNA_output_high_finetune.png"  width="200"> <img src= "./media/chat_Enhancer.png" width="200"> 


### Severel remarks 

* Current backbone diffusion models for fine-tuning are Dirichlet Diffusion Models in  [Avedeyev et.al, 2023](https://arxiv.org/abs/2305.10699) (We acknowledge that we borrow the code from their code in some parts. We will add more diffusion models tailoed to sequences. )

* Is overoptimization happening?: Check it out [Uehera and Zhao et.al, 2024](https://arxiv.org/abs/2405.19673) on how to avodi it.

* Lab in the loop? : Chekc it out Yulai's implementation here https://github.com/zhaoyl18/SEIKO and our paper[Uehara and Zhao et.al, 2024](https://arxiv.org/abs/2402.16359)

### Acknowledgement
* Reward models are based on the enfomrer-model [(Avset et al., 2021)](https://www.nature.com/articles/s41592-021-01252-x), one of the most common models for DNA sequence modeling. We use Grelu package for this purpose. 
* Enahcer dataset: Originally, it is from [Gosai et al., 2023](https://www.biorxiv.org/content/10.1101/2023.08.08.552077v1)
* UTR dataset: Originally, it is from  [Sample et al., 2019 ](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE114002)

### Installation 

If we have GRELU package (https://github.com/Genentech/gReLU) (+ standard packages) work. 
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install grelu tqdm pandas matplotlib lightning
```

### Citation

If you find this work useful in your research, please cite: