import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import sys,os
from . import ddsm

class Dense(nn.Module):
    """
    A fully connected layer that reshapes outputs to feature maps.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.dense(x)[...]

    
class ScoreNet(nn.Module):
  """A time-dependent score-based model built upon U-Net architecture."""

  def __init__(self, embed_dim=256, time_dependent_weights=None, class_number = None, time_step=0.01):
    """
    Initialize a time-dependent score-based network.
    Args:
      marginal_prob_std: A function that takes time t and gives the standard
      deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
      channels: The number of channels for feature maps of each resolution.
      embed_dim: The dimensionality of Gaussian random feature embeddings.
    """
    super().__init__()
    # Gaussian random feature embedding layer for time
    self.embed = nn.Sequential(ddsm.GaussianFourierProjection(embed_dim=embed_dim),
         nn.Linear(embed_dim, embed_dim))
    
    n=256
    self.linear = nn.Conv1d(4, n, kernel_size=9, padding=4)
    self.blocks = nn.ModuleList([nn.Conv1d(n, n, kernel_size=9, padding=4),
                nn.Conv1d(n, n, kernel_size=9, padding=4),
                nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
                nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
                nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256),
                nn.Conv1d(n, n, kernel_size=9, padding=4),
                nn.Conv1d(n, n, kernel_size=9, padding=4),
                nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
                nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
                nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256),
                nn.Conv1d(n, n, kernel_size=9, padding=4),
                nn.Conv1d(n, n, kernel_size=9, padding=4),
                nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
                nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
                nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256),
                nn.Conv1d(n, n, kernel_size=9, padding=4),
                nn.Conv1d(n, n, kernel_size=9, padding=4),
                nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
                nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
                nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256)])



    self.denses = nn.ModuleList([Dense(embed_dim, n) for _ in range(20)])
    self.norms = nn.ModuleList([nn.GroupNorm(1,n) for _ in range(20)])

    # The swish activation function
    self.act = lambda x: x * torch.sigmoid(x)
    self.relu = nn.ReLU()
    self.softplus = nn.Softplus()
    self.scale = nn.Parameter(torch.ones(1))
    self.final =  nn.Sequential(nn.Conv1d(n, n, kernel_size=1),
                                nn.GELU(),
                                nn.Conv1d(n, 4, kernel_size=1))
    self.register_buffer("time_dependent_weights", time_dependent_weights)
    self.time_step = time_step
    
  def forward(self, x, t, class_number = None, t_ind=None, return_a=False):
    # Obtain the Gaussian random feature embedding for t
    # embed: [N, embed_dim]
    embed = self.act(self.embed(t/2))

    # Encoding path
    # x: NLC -> NCL
    out = x.permute(0, 2, 1)
    out = self.act(self.linear(out))
    
    #pos encoding
    for block, dense, norm in zip(self.blocks, self.denses, self.norms):
        h = self.act(block(norm(out + dense(embed)[:,:,None])))
        if h.shape == out.shape:
            out = h + out
        else:
            out = h

    
    out = self.final(out)
    out = out.permute(0, 2, 1)
    if self.time_dependent_weights is not None:
        t_step = (t/self.time_step)-1
        w0 = self.time_dependent_weights[t_step.long()]
        w1 = self.time_dependent_weights[torch.clip(t_step+1,max=len(self.time_dependent_weights)-1).long()]
        out = out * (w0 + (t_step-t_step.floor())*(w1-w0))[:,None,None]

    out = out - out.mean(axis=-1, keepdims=True)
    return out


class ScoreNet_Conditional(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, embed_dim=256, time_dependent_weights=None, all_class_number = 2, time_step=0.01):
        """Initialize a time-dependent score-based network.

        Args:
          marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
          channels: The number of channels for feature maps of each resolution.
          embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(ddsm.GaussianFourierProjection(embed_dim=embed_dim),
                                   nn.Linear(embed_dim, embed_dim))
        n = 256
        self.linear = nn.Conv1d(4, n, kernel_size=9, padding=4)
        self.blocks = nn.ModuleList([nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256)])

        self.denses = nn.ModuleList([Dense(embed_dim, n) for _ in range(20)])
        self.norms = nn.ModuleList([nn.GroupNorm(1, n) for _ in range(20)])
        self.cls_layers = nn.ModuleList([ Dense(embed_dim, embed_dim) for _ in range(20)] )
        self.embed_class = nn.Embedding(num_embeddings= all_class_number+1, embedding_dim= embed_dim)
        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.scale = nn.Parameter(torch.ones(1))
        self.final = nn.Sequential(nn.Conv1d(n, n, kernel_size=1),
                                   nn.GELU(),
                                   nn.Conv1d(n, 4, kernel_size=1))
        self.register_buffer("time_dependent_weights", time_dependent_weights)
        self.time_step = time_step
       
    def forward(self, x, t, class_number = None, t_ind=None, return_a=False):
        # Obtain the Gaussian random feature embedding for t
        # embed: [N, embed_dim]

        embed = self.act(self.embed(t / 2))
        cond_embed = self.embed_class(class_number)

        # Encoding path
        # x: NLC -> NCL
        out = x.permute(0, 2, 1)
        out = self.act(self.linear(out))

        # pos encoding
        for block, dense, norm, cond_class in zip(self.blocks, self.denses, self.norms, self.cls_layers):
            h = self.act(block(norm(out + dense(embed)[:, :, None] + cond_class(cond_embed) [:, :, None] )))
          
            if h.shape == out.shape:
                out = h + out 
            else:
                out = h 

        out = self.final(out)

        out = out.permute(0, 2, 1)

        if self.time_dependent_weights is not None:
            t_step = (t / self.time_step) - 1
            w0 = self.time_dependent_weights[t_step.long()]
            w1 = self.time_dependent_weights[torch.clip(t_step + 1, max=len(self.time_dependent_weights) - 1).long()]
            out = out * (w0 + (t_step - t_step.floor()) * (w1 - w0))[:, None, None]

        out = out - out.mean(axis=-1, keepdims=True)
        return out
    
