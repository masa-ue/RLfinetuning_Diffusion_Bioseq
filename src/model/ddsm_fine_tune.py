import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import one_hot
from . import ddsm
import numpy as np

import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

def Euler_Maruyama_sampler_GPU_Conditional(
        score_model,
        original_model,
        sample_shape,
        existing_condition,
        additional_condition,
        strength,
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
        gradient_start = 90, 
        device="cuda",
        random_order=False,
        speed_balanced=True,
        speed_factor=None,
        concat_input=None,
        eps=1e-5,
        continuous=True
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
    
    entropy = torch.zeros(batch_size).to(device)
    
    #with torch.no_grad():
    for i_step in range(len(time_steps)):
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
                    # score = score_model(x, batch_time_step, condition, additional_condition, None) # x: [128, 50, 4]; batch_time_step: [128], condition: [128, 1]
                    if continuous:
                        zero_class = torch.zeros(batch_size).type(torch.FloatTensor).to(device)
                    else:
                        zero_class = torch.zeros(batch_size).type(torch.LongTensor).to(device)
                    
                    score = strength * score_model(x, batch_time_step, existing_condition, additional_condition) +\
                        (1.0 - strength) * score_model(x, batch_time_step, zero_class, zero_class) 
                else:
                    # score = score_model(torch.cat([x, concat_input], -1), batch_time_step, 
                    #         existing_condition, additional_condition, None)
                    raise NotImplementedError

                if i_step <= gradient_start:
                    transform = ddsm.gx_to_gv(score, x)
                else:
                    transform = ddsm.gx_to_gv(score, x, True) 
                    original_score = original_model(x, batch_time_step, existing_condition, None)
                    original_transform = ddsm.gx_to_gv(original_score, x)  
                drift =  (
                        (0.5 * (alpha[(None,) * (v.ndim - 1)] * (1 - v)
                                - beta[(None,) * (v.ndim - 1)] * v)) - (1 - 2 * v)
                        - (g ** 2) * transform  ) * (-step_size) * c * s[(None,) * (v.ndim - 1)]  
                mean_v = v + drift  

                # Entropy Calcuation
                if i_step > gradient_start: 
                    drift_difference = ((g * g* g) * (original_transform - transform) * (original_transform - transform) ) * (step_size) * c * s[(None,) * (v.ndim - 1)]  
                    entropy = entropy + 0.5 * torch.sum(drift_difference, [1,2])

                next_v = mean_v + torch.sqrt(step_size * c) * \
                        torch.sqrt(s[(None,) * (v.ndim - 1)]) * g * torch.randn_like(v)

                if mask is not None:
                    next_v[~torch.isnan(mask_v)] = mask_v[~torch.isnan(mask_v)]

                v = torch.clamp(next_v, eps, 1 - eps)#.detach()
        

    if mask is not None:
        mean_v[~torch.isnan(mask_v)] = mask_v[~torch.isnan(mask_v)]

    # Do not include any noise in the last sampling step.
    if not random_order:
        return sb(torch.clamp(mean_v, eps, 1 - eps)), entropy
    else:
        return sb(torch.clamp(mean_v, eps, 1 - eps))[..., np.argsort(order)]



def fine_tuning(
            score_model, 
            reward_model, 
            eval_model, 
            original_model,
            learning_rate=5*1e-5, 
            num_epoch=100, 
            num_steps=50, 
            accmu=4, 
            length=200, 
            gradient_start=45, 
            batch_size=32,
            guidance_strength=5.0, 
            max_time=4.0, 
            min_time=1/400, 
            entropy_coff=5 * 1e-6, 
            speed_balanced=True, 
            save_folder="log/", 
            device='cuda'
        ):
    
    # reward_list = [ ]
    # num_eval = len(eval_model)
    # eval_list =[ [] for i in range(num_eval)]
    
    eval_pearson_old = []
    eval_pearson_new = []
    
    for k in range(num_epoch): 
        torch.set_grad_enabled(True)
        optim = torch.optim.Adam(score_model.parameters(), lr=learning_rate)
        
        existing_condition = torch.empty(batch_size, device=device).uniform_(-1, 1)
        additional_condition = torch.empty(batch_size, device=device).uniform_(-1, 1)
        
        # Training 
        logits_pred, entropy = Euler_Maruyama_sampler_GPU_Conditional(
                                        score_model, 
                                        original_model,
                                        (length,4),
                                        existing_condition=existing_condition,
                                        additional_condition=additional_condition,
                                        strength=guidance_strength,
                                        batch_size=batch_size,
                                        max_time=max_time,
                                        min_time=min_time,
                                        time_dilation=1,
                                        num_steps=num_steps, 
                                        gradient_start=gradient_start,
                                        eps=1e-5,
                                        speed_balanced=speed_balanced,
                                        device=device
                                    )
        
        raw_reward = reward_model(logits_pred)[:,1].squeeze(1) #### logits_pred: [128, 50, 4] -> reward: [128,3,1]
        cls_reward = -torch.square(raw_reward - additional_condition)
        # probabilities = reward_model(logits_pred).squeeze(-1)
        # selected_probs = probabilities[torch.arange(probabilities.size(0)), existing_condition]
        # reward = torch.log(selected_probs + 1e-6)
        
        loss = -torch.mean(cls_reward - entropy_coff * entropy)/accmu
        entropy_accum = torch.mean(entropy)/accmu
        
        pearson_k562, _ = pearsonr(raw_reward.detach().cpu().numpy(), 
                        additional_condition.cpu().numpy())
        
        print('Epoch: %d | Loss: %.2f | entropy: %.2f | K562 Pearson %.2f \n'%(k, loss.item(), entropy_accum.item(), pearson_k562))

        loss.backward()
        
        ### Update (Gradient accumulation)
        if  (k+1) % accmu == 0:
            optim.step()
            optim.zero_grad()

            torch.save(score_model.state_dict(), save_folder +"/%d.pth"%k)

            # Evaluation
            total_samples = batch_size * 2
                       
            # samples_per_class = [int(total_samples * 0.5), 0, int(total_samples * 0.5)]
            # eval_condition = torch.cat([torch.full((count,), i, device=device, dtype=torch.long) for i, count in enumerate(samples_per_class)])
            
            eval_existing_condition = torch.empty(total_samples, device=device).uniform_(-1, 4)
            eval_additional_condition = torch.empty(total_samples, device=device).uniform_(-1, 4)
            
            with torch.no_grad():
                logits_pred  = ddsm.Euler_Maruyama_sampler(
                                    score_model,
                                    (length,4), 
                                    existing_condition = eval_existing_condition,
                                    additional_condition = eval_additional_condition,
                                    class_number=0,
                                    strength=guidance_strength,
                                    batch_size=total_samples,
                                    max_time=max_time,
                                    min_time=min_time,
                                    time_dilation=1,
                                    num_steps=num_steps, 
                                    eps=1e-5,
                                    speed_balanced=speed_balanced,
                                    device= device
                                )
                
                # reward = torch.mean(reward_model(logits_pred)[:,1]) #### Evaluate_reward 
                
                raw_reward_old_condition = reward_model(logits_pred)[:,0].squeeze(1)
                raw_reward_new_condition = reward_model(logits_pred)[:,1].squeeze(1)

                pearson_old, _ = pearsonr(raw_reward_old_condition.detach().cpu().numpy(), eval_existing_condition.cpu().numpy())
                pearson_new, _ = pearsonr(raw_reward_new_condition.detach().cpu().numpy(), eval_additional_condition.cpu().numpy())
                
            print('Evaluation ------ HepG2 Pearson: %.3f | K562 Pearson: %.3f\n'%(pearson_old, pearson_new))
            
            eval_pearson_old.append(pearson_old)
            eval_pearson_new.append(pearson_new)
            
            # probabilities = reward_model(logits_pred).squeeze(-1)
            # selected_probs = probabilities[torch.arange(probabilities.size(0)), eval_condition]
            # reward = torch.mean(torch.log(selected_probs + 1e-6))
            
            # reward_list.append([k, reward.item()])
            # print(reward.item())

            # with torch.no_grad():
            #     for i, func in enumerate(eval_model): 
            #         reward = torch.mean(func(logits_pred))
            #         # probabilities = func(logits_pred).squeeze(-1)
            #         # selected_probs = probabilities[torch.arange(probabilities.size(0)), eval_condition]
            #         # reward = torch.mean(selected_probs)
            
            #         eval_list[i].append(reward.item()) 
            
                
    return eval_pearson_old, eval_pearson_new


 