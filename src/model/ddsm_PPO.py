import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import one_hot
import numpy as np
import tqdm
import wandb
from . import ddsm 


def Euler_Maruyama_sampler_PPO(
        score_model,
        origianl_model,
        sample_shape,
        init=None,
        mask=None,
        alpha=None,
        beta=None,
        max_time=4,
        min_time=0.01,
        time_dilation=1,
        time_dilation_start_time=None,
        batch_size=64,
        num_steps=100,
        device="cuda",
        random_order=False,
        speed_balanced=True,
        speed_factor=None,
        concat_input=None,
        eps=1e-5,
):
    """
    Generate samples from score-based models with the Euler-Maruyama solver
    for (multivariate) Jacobi diffusion processes with stick-breaking
    construction.

    Parameters
    ----------
    score_model : torch.nn.Module
        A PyTorch time-dependent score model.
    sample_shape : tuple
        Shape of all dimensions of sample tensor without the batch dimension.
    init: torch.Tensor, default is None
        If specified, use as initial values instead of sampling from stationary distribution.
    alpha :  torch.Tensor, default is None
        Jacobi Diffusion parameters. If None, use default choices of alpha, beta =
        (1, k-1), (1, k-2), (1, k-3), ..., (1, 1) where k is the number of categories.
    beta : torch.Tensor, default is None
        See above `for alpha`.
    max_time : float, default is 4
        Max time of reverse diffusion sampling.
    min_time : float, default is 0.01
        Min time of reverse diffusion sampling.
    time_dilation : float, default is 1
        Use `time_dilation > 1` to bias samples toward high density areas.
    time_dilation_start_time : float, default is None
        If specified, start time dilation from this timepoint to min_time.
    batch_size : int, default is 64
        Number of samples to generate
    num_steps: int, default is 100
        Total number of steps for reverse diffusion sampling.
    device: str, default is 'cuda'
        Use 'cuda' to run on GPU or 'cpu' to run on CPU
    random_order : bool, default is False
        Whether to convert x to v space with randomly ordered stick-breaking transform.
    speed_balanced : bool, default is True
        If True use speed factor `s=(a+b)/2`, otherwise use `s=1`.
    eps: float, default is 1e-5
        All state values are clamped to (eps, 1-eps) for numerical stability.


    Returns
    -------
    Samples : torch.Tensor
        Samples in x space.
    """
    sb = ddsm.UnitStickBreakingTransform()
    if alpha is None:
        alpha = torch.ones(sample_shape[-1] - 1, dtype=torch.float, device=device)
    if beta is None:
        beta = torch.arange(
            sample_shape[-1] - 1, 0, -1, dtype=torch.float, device=device
        )

    if speed_balanced:
        if speed_factor is None:
            s = 2.0 / (alpha + beta)
        else:
            s = speed_factor * 2.0 / (alpha + beta)
    else:
        s = torch.ones(sample_shape[-1] - 1).to(device)

    if init is None:
        init_v = ddsm.Beta(alpha, beta).sample((batch_size,) + sample_shape[:-1]).to(device)
    else:
        init_v = sb._inverse(init).to(device)

    if time_dilation_start_time is None:
        time_steps = torch.linspace(
            max_time, min_time, num_steps * time_dilation + 1, device=device
        )
    else:
        time_steps = torch.cat(
            [
                torch.linspace(
                    max_time,
                    time_dilation_start_time,
                    round(num_steps * (max_time - time_dilation_start_time) / max_time)
                    + 1,
                )[:-1],
                torch.linspace(
                    time_dilation_start_time,
                    min_time,
                    round(num_steps * (time_dilation_start_time - min_time) / max_time)
                    * time_dilation
                    + 1,
                ),
            ]
        )
    step_sizes = time_steps[:-1] - time_steps[1:]
    time_steps = time_steps[:-1]
    v = init_v#.detach()


    if mask is not None:
        assert mask.shape[-1] == v.shape[-1]+1

    if random_order:
        order = np.arange(sample_shape[-1])
    else:
        if mask is not None:
            mask_v = sb.inv(mask)
    


    with torch.no_grad():

        old_noise_list = [ ]
        v_list = []
        var_list = []
        mean_v_list = []

        for i_step in tqdm.tqdm(range(len(time_steps))):
            time_step = time_steps[i_step]
            step_size = step_sizes[i_step]
            x = sb(v)
            
            if time_dilation_start_time is not None:
                if time_step < time_dilation_start_time:
                    c = time_dilation
                else:
                    c = 1
            else:
                c = time_dilation

            if not random_order:
                
                
                g = torch.sqrt(v * (1 - v))
                batch_time_step = torch.ones(batch_size, device=device) * time_step
                with torch.enable_grad():
                    if concat_input is None:
                        score = score_model(x, batch_time_step, None)
                    else:
                        score = score_model(torch.cat([x, concat_input], -1), batch_time_step, None)

                    transform = ddsm.gx_to_gv(score, x)
                    #origianl_score = origianl_model(x, batch_time_step, None)
                    #original_transform = ddsm.gx_to_gv(origianl_score, x)  
                    drift =  (
                            (0.5 * (alpha[(None,) * (v.ndim - 1)] * (1 - v)
                                    - beta[(None,) * (v.ndim - 1)] * v)) - (1 - 2 * v)
                            - (g ** 2) * transform  ) * (-step_size) * c * s[(None,) * (v.ndim - 1)]  
                mean_v = v + drift  

                # Entropy Calcuation
                #if i_step > gradient_start: 
                #    drift_difference = ((g * g* g) * (original_transform - transform) * (original_transform - transform) ) * (step_size) * c * s[(None,) * (v.ndim - 1)]  
                #    entropy = entropy + 0.5 * torch.sum(drift_difference, [1,2])
                
                old_noise = torch.randn_like(v) 
                
                
                old_noise_list.append(old_noise)
                var = torch.sqrt(step_size * c) *   torch.sqrt(s[(None,) * (v.ndim - 1)]) * g 
                
                var_list.append(var)
                v_list.append(v)
                mean_v_list.append(mean_v)

                next_v = mean_v + var * old_noise

                if mask is not None:
                    next_v[~torch.isnan(mask_v)] = mask_v[~torch.isnan(mask_v)]
                v = torch.clamp(next_v, eps, 1 - eps).detach()
            

        if mask is not None:
            mean_v[~torch.isnan(mask_v)] = mask_v[~torch.isnan(mask_v)]

        # Do not include any noise in the last sampling step.
        if not random_order:
            return sb(torch.clamp(mean_v, eps, 1 - eps)), old_noise_list, v_list, var_list, mean_v_list
        else:
            return sb(torch.clamp(mean_v, eps, 1 - eps))[..., np.argsort(order)]


def calculate_newprob( 
        given_v, i_step, 
        score_model,
        origianl_model,
        sample_shape,
        init=None,
        mask=None,
        alpha=None,
        beta=None,
        max_time=4,
        min_time=0.01,
        time_dilation=1,
        time_dilation_start_time=None,
        batch_size=64,
        num_steps=100,
        device="cuda",
        random_order=False,
        speed_balanced=True,
        speed_factor=None,
        concat_input=None,
        eps=1e-5,
):
    sb = ddsm.UnitStickBreakingTransform()
    if alpha is None:
        alpha = torch.ones(sample_shape[-1] - 1, dtype=torch.float, device=device)
    if beta is None:
        beta = torch.arange(
            sample_shape[-1] - 1, 0, -1, dtype=torch.float, device=device
        )

    if speed_balanced:
        if speed_factor is None:
            s = 2.0 / (alpha + beta)
        else:
            s = speed_factor * 2.0 / (alpha + beta)
    else:
        s = torch.ones(sample_shape[-1] - 1).to(device)

    if time_dilation_start_time is None:
        time_steps = torch.linspace(
            max_time, min_time, num_steps * time_dilation + 1, device=device
        )
    else:
        time_steps = torch.cat(
            [
                torch.linspace(
                    max_time,
                    time_dilation_start_time,
                    round(num_steps * (max_time - time_dilation_start_time) / max_time)
                    + 1,
                )[:-1],
                torch.linspace(
                    time_dilation_start_time,
                    min_time,
                    round(num_steps * (time_dilation_start_time - min_time) / max_time)
                    * time_dilation
                    + 1,
                ),
            ]
        )
    step_sizes = time_steps[:-1] - time_steps[1:]
    time_steps = time_steps[:-1]
    
    v = given_v 

    if mask is not None:
        assert mask.shape[-1] == v.shape[-1]+1
    
    time_step = time_steps[i_step]
    step_size = step_sizes[i_step]
    x = sb(v)
    
    if time_dilation_start_time is not None:
        if time_step < time_dilation_start_time:
            c = time_dilation
        else:
            c = 1
    else:
        c = time_dilation

    if not random_order:
        
        g = torch.sqrt(v * (1 - v))
        batch_time_step = torch.ones(batch_size, device=device) * time_step

        if concat_input is None:
            score = score_model(x, batch_time_step, None)
        else:
            score = score_model(torch.cat([x, concat_input], -1), batch_time_step, None)

        transform = ddsm.gx_to_gv(score, x, True)
        drift =  (
                (0.5 * (alpha[(None,) * (v.ndim - 1)] * (1 - v)
                        - beta[(None,) * (v.ndim - 1)] * v)) - (1 - 2 * v)
                - (g ** 2) * transform  ) * (-step_size) * c * s[(None,) * (v.ndim - 1)]  
        mean_v = v + drift  

        return mean_v


def fine_tuning(score_model, reward_model, eval_model, original_model, learning_rate = 5*1e-5, num_epoch = 100, num_steps = 50, accmu = 4, length = 200, batch_size = 32 , max_time = 4.0, min_time = 1/400, entropy_coff = 5 * 1e-6, speed_balanced = True, save_name = "trained.pth", device ='cuda' ):
    
    reward_list = [ ]
    num_eval = len(eval_model)
    eval_list =[ [] for i in range(num_eval)]
    wandb.login(host="https://api.wandb.ai") 
    run = wandb.init()

    optim = torch.optim.Adam(score_model.parameters(), lr=learning_rate)

    batch_losses = []
    for k in range(num_epoch): 
        # Here No GRAD 
        with torch.no_grad():
            # Training 
            logits_pred, old_noise_list, v_list, var_list, mean_v_list = Euler_Maruyama_sampler_PPO(score_model, original_model,
                                (length,4),
                                batch_size= batch_size ,
                                max_time= max_time,
                                min_time= min_time,
                                time_dilation=1,
                                num_steps= num_steps, 
                                eps=1e-5,
                                speed_balanced= speed_balanced,
                                device= device
                                )
            reward = reward_model(logits_pred) #### Evaluate_reward

     

        ratio_list = []
        for random_t in range(num_steps):
            # Calculate pi_{old}(a)
            with torch.no_grad(): 
                intem_old = -torch.square(old_noise_list[random_t]) * 0.5 
                log_old_probability = torch.sum(intem_old, dim = (1,2) )
                
            # Calculate pi_{new}(a)
            torch.set_grad_enabled(True)

            next_mean  = calculate_newprob(v_list[random_t], random_t, score_model, original_model,
                                (length,4),
                                batch_size= batch_size ,
                                max_time= max_time,
                                min_time= min_time,
                                time_dilation=1,
                                num_steps= num_steps, 
                                eps=1e-5,
                                speed_balanced= speed_balanced,
                                device= device
                                )
            
            intem_new = -torch.square(old_noise_list[random_t] + (next_mean - mean_v_list[random_t])/var_list[random_t]  ) * 0.5 
            log_new_probability = torch.sum(intem_new, dim = (1,2) )

            # Calculate pi_{new}(a)/pi_{old}(a)
            ratio = torch.exp(log_new_probability-log_old_probability) 
            ratio_list.append(ratio)

   
        batch_losses.append(torch.mean(reward).cpu().detach().numpy())
        loss =0.0
        for random_t in range(num_steps): 
            ratio = ratio_list[random_t]
            loss = loss  - torch.mean(torch.min(reward * ratio, reward * torch.clamp(ratio , min = 1.0 - 0.2, max = 1.0 + 0.2) )) * 1.0/(num_steps * accmu)

        # Update parameters 
        loss.backward()

        # Evaluation 
        if  (k+1) % accmu ==0:
            optim.step()
            optim.zero_grad()
            torch.save(score_model.state_dict(), save_name +"_%d.pth"%k)
            # Evaluation 
            logits_pred  = ddsm.Euler_Maruyama_sampler(score_model,
                            (length,4), 
                            batch_size= batch_size*4 ,
                            new_class = None,
                            class_number = 1,
                            max_time= max_time,
                            min_time= min_time,
                            time_dilation=1,
                            num_steps= num_steps, 
                            eps=1e-5,
                            speed_balanced= speed_balanced,
                            device= device
                            )
            reward = torch.mean(reward_model(logits_pred)) #### Evaluate_reward 
            reward_list.append([k,reward.item()])
            print(reward.item())
            run.log({"reward": reward.item()} ) 
            with torch.no_grad():
                for i, func in enumerate(eval_model): 
                    reward = torch.mean(func(logits_pred))
                    eval_list[i].append(reward.item()) 
    wandb.finish()
    return reward_list, eval_list





if __name__ == "__main__":
    import ddsm as ddsm
    from lightning_model_diffusion import LightningDiffusion as lightning_dif
    import os
    save_name = "log_MPRA2/"
    isExist = os.path.exists(save_name)
    if not isExist:
        os.makedirs(save_name)
     
    class ModelParameters:
        diffusion_weights_file = '/home/ueharam1/projects3/diffusion-dna-rna-main/tutorials/UTR/artifacts/UTR-dataset:v0/steps400.cat4.speed_balance.time4.0.samples100000.pth'
        time_schedule = "/home/ueharam1/projects3/diffusion-dna-rna-main/tutorials/UTR/artifacts/UTR-dataset:v0/time_dependent.npz"
        checkpoint_path = '/home/ueharam1/projects3/diffusion-dna-rna-main/tutorials/UTR/artifacts/UTR-Model:v0/diffusion_unconditional_epoch=075.ckpt'

    config = ModelParameters() 
    DEVICE = "cuda:2" # Any number is fine

    # Introduce Two Models
    score_model = lightning_dif.load_from_checkpoint(checkpoint_path= config.checkpoint_path, weight_file = config.diffusion_weights_file, time_schedule = config.time_schedule, all_class_number =1)
    score_model = score_model.model
    score_model.cuda(device = DEVICE) 


    original_model = lightning_dif.load_from_checkpoint(checkpoint_path= config.checkpoint_path, weight_file = config.diffusion_weights_file, time_schedule = config.time_schedule, all_class_number =1)
    original_model = original_model.model
    original_model.cuda(device = DEVICE) 
    # Load Reward model

    from grelu.lightning import LightningModel
    model = LightningModel.load_from_checkpoint("/home/ueharam1/projects3/diffusion-dna-rna-main/tutorials/UTR/artifacts/UTR-model/reward_model.ckpt")
    model.eval()
    model.to(DEVICE)

    def new_reward_model(x):
        x = torch.nn.functional.softmax(x /0.1, -1)
        seq = torch.transpose(x, 1, 2) 
        return model(seq)


    loss_curves, eval_curves = fine_tuning(score_model, new_reward_model, [new_reward_model], original_model,
            learning_rate =5e-3, num_epoch = 1000, length = 50, num_steps = 50, accmu = 4,
            batch_size = 128, save_name = save_name, entropy_coff = 0.0,  device= DEVICE)