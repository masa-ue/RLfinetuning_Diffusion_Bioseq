import torch
import torch.nn as nn
import torch.nn.functional as F
from . import ddsm
import torch.nn.init as init


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

    def __init__(self, embed_dim=256, time_dependent_weights=None, all_class_number=3, time_step=0.01):
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
        
        self.embed_class = nn.Embedding(num_embeddings=all_class_number+1, 
                                        embedding_dim=embed_dim)
        
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
       
    def forward(self, x, t, class_number=None, t_ind=None, return_a=False):
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
            h = self.act(block(norm(out + dense(embed)[:, :, None] + cond_class(cond_embed)[:, :, None] ))) # out:[128, 256, 50], dense(embed)[:, :, None]: [128, 256, 1], cond_class(cond_embed)[:, :, None]: [128, 256, 1]     
          
            if h.shape == out.shape:  # h.shape: [128, 256, 50], out.shape: [128, 256, 50]
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
        return out # torch.Size([128, 50, 4])
    

class AugmentedScoreNet_Conditional(ScoreNet_Conditional):
    def __init__(self, embed_dim=256, time_dependent_weights=None, all_class_number=3, time_step=0.01, augment=False):
        super().__init__(embed_dim, time_dependent_weights, all_class_number, time_step)
        
        if augment:
            self.additional_embed_class = nn.Embedding(num_embeddings=all_class_number+1, embedding_dim=embed_dim)
            self.additional_embed_class.weight.data.zero_()
        else:
            self.additional_embed_class = None

    def forward(self, x, t, existing_condition=None, add_condition=None, t_ind=None, return_a=False):
        # Call the original forward method
        embed = self.act(self.embed(t / 2))
        cond_embed = self.embed_class(existing_condition)
        
        if add_condition is not None:
            assert self.additional_embed_class is not None
            add_con_embed = self.additional_embed_class(add_condition)
        else:
            add_con_embed = torch.zeros_like(cond_embed)

        # Encoding path
        out = x.permute(0, 2, 1)
        out = self.act(self.linear(out))

        # pos encoding
        for block, dense, norm, cond_class in zip(self.blocks, self.denses, self.norms, self.cls_layers):
            h = self.act(block(norm(out + 
                                    dense(embed)[:, :, None] + 
                                    cond_class(cond_embed)[:, :, None] + 
                                    cond_class(add_con_embed)[:, :, None]
                                    )
                               )
                         )  
          
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



class ScoreNet_Continuous(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, 
                 embed_dim=256, 
                 time_dependent_weights=None, 
                 y_low=0, 
                 y_high=10, 
                 time_step=0.01
            ):
        """Initialize a time-dependent score-based network.

        Args:
          marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
          channels: The number of channels for feature maps of each resolution.
          embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        super().__init__()
        
        self.y_low = y_low
        self.y_high = y_high
        
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
        
        self.embed_class = nn.Sequential(ddsm.GaussianFourierProjection(embed_dim=embed_dim, scale=40.0),
                                         nn.Linear(embed_dim, embed_dim))
        
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
       
    def forward(self, x, t, class_number=None, t_ind=None, return_a=False):
        # Obtain the Gaussian random feature embedding for t
        # embed: [N, embed_dim]
        
        normalized_class_value = (class_number - self.y_low) / (self.y_high - self.y_low) # Normalize to (0,1)

        embed = self.act(self.embed(t / 2))
        cond_embed = self.act(self.embed_class(normalized_class_value))

        # Encoding path
        # x: NLC -> NCL
        out = x.permute(0, 2, 1)
        out = self.act(self.linear(out))

        # pos encoding
        for block, dense, norm, cond_class in zip(self.blocks, self.denses, self.norms, self.cls_layers):
            h = self.act(block(norm(out + dense(embed)[:, :, None] + cond_class(cond_embed)[:, :, None] ))) # out:[128, 256, 50], dense(embed)[:, :, None]: [128, 256, 1], cond_class(cond_embed)[:, :, None]: [128, 256, 1]     
          
            if h.shape == out.shape:  # h.shape: [128, 256, 50], out.shape: [128, 256, 50]
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
        return out # torch.Size([128, 50, 4])
    


class AugmentedScoreNet_Continuous(ScoreNet_Continuous):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, 
                 embed_dim=256, 
                 time_dependent_weights=None, 
                 y_low=0, 
                 y_high=10, 
                 augment=False,
                 time_step=0.01
                ):
        """Initialize a time-dependent score-based network.

        Args:
          marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
          channels: The number of channels for feature maps of each resolution.
          embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        super().__init__()
        
        self.augment = augment
        
        if self.augment:
            self.additional_embed = nn.Sequential(ddsm.GaussianFourierProjection(embed_dim=embed_dim),
                                   nn.Linear(embed_dim, embed_dim))
            init.zeros_(self.additional_embed[1].weight)
            init.zeros_(self.additional_embed[1].bias)
        else:
            self.additional_embed = None

       
    def forward(self, x, t, class_number=None, add_class_number=None, t_ind=None, return_a=False):
        # Obtain the Gaussian random feature embedding for t
        # embed: [N, embed_dim]
        
        normalized_class_value = (class_number - self.y_low) / (self.y_high - self.y_low) # Normalize to (0,1)

        embed = self.act(self.embed(t / 2))
        cond_embed = self.act(self.embed_class(normalized_class_value))

        if add_class_number is not None and self.additional_embed is not None:
            normalized_add_class_value = (add_class_number - self.y_low) / (self.y_high - self.y_low)
            add_con_embed = self.additional_embed(normalized_add_class_value)
        else:
            add_con_embed = torch.zeros_like(cond_embed)
        
        # Encoding path
        # x: NLC -> NCL
        out = x.permute(0, 2, 1)
        out = self.act(self.linear(out))

        # pos encoding
        for block, dense, norm, cond_class in zip(self.blocks, self.denses, self.norms, self.cls_layers):
            h = self.act(block(norm(out + 
                                    dense(embed)[:, :, None] + 
                                    cond_class(cond_embed)[:, :, None] +
                                    cond_class(add_con_embed)[:, :, None]
                                    ))) # out:[128, 256, 50], dense(embed)[:, :, None]: [128, 256, 1], cond_class(cond_embed)[:, :, None]: [128, 256, 1]     
          
            if h.shape == out.shape:  # h.shape: [128, 256, 50], out.shape: [128, 256, 50]
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
        return out # torch.Size([128, 50, 4])
    
