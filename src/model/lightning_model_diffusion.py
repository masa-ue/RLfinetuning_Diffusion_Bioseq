import torch
import model.ddsm as ddsm 
import model.ddsm_model as modeld
import numpy as np
import wandb
import lightning as L
from torch.optim import Adam

sb = ddsm.UnitStickBreakingTransform()


class LightningDiffusion(L.LightningModule):
    def __init__(self, weight_file, time_schedule, speed_balanced = True, all_class_number = 2 , ncat = 4, n_time_steps = 400, lr = 1e-4):

        super().__init__() 

        v_one, v_zero, v_one_loggrad, v_zero_loggrad, timepoints = torch.load(weight_file)

        # Change to CPU  
        self.v_one = v_one.cpu()
        self.v_zero = v_zero.cpu()
        self.time_schedule = time_schedule
        self.v_one_loggrad = v_one_loggrad.cpu()
        self.v_zero_loggrad = v_zero_loggrad.cpu()
        self.timepoints = timepoints.cpu()
        
        time_dependent_weights = torch.tensor(np.load(time_schedule)['x'])
        if all_class_number != 1:
            self.model = modeld.ScoreNet_Conditional(time_dependent_weights=torch.sqrt(time_dependent_weights), all_class_number= all_class_number)
        else:
            self.model = modeld.ScoreNet(time_dependent_weights=torch.sqrt(time_dependent_weights)) 
        self.avg_loss = 0  
        self.num_items = 0
        self.all_class_number = all_class_number
        self.speed_balanced = speed_balanced
        self.ncat = ncat
        self.n_time_steps = n_time_steps
        self.lr = lr

    def forward(self, x, t, y):
        return self.model(x, t, class_number = y)
    
    def train_epoch_start(self):
        self.num_items = 0
        self.avg_loss = 0 

    def training_step(self, batch):
        xS, yS = batch
        x = xS[:, :, :4]
        time_dependent_weights = torch.tensor(np.load(self.time_schedule)['x']).to(self.device)
 
        # Optional : there are several options for importance sampling here. it needs to match the loss function
        random_t = torch.LongTensor(np.random.choice(np.arange(self.n_time_steps), size=x.shape[0],
                                                     p=(torch.sqrt(time_dependent_weights) / torch.sqrt(
                                                         time_dependent_weights).sum()).cpu().detach().numpy()))
       
        perturbed_x, perturbed_x_grad = ddsm.diffusion_fast_flatdirichlet(x.cpu(), random_t, self.v_one, self.v_one_loggrad)
  
        perturbed_x = perturbed_x.to(self.device)
        perturbed_x_grad = perturbed_x_grad.to(self.device)
        random_timepoints = self.timepoints[random_t].to(self.device)
        
        yS = yS.type(torch.LongTensor)
        random_list = np.random.binomial(1,0.3, yS.shape[0])
        yS[random_list ==1 ] = self.all_class_number
        yS = yS.to(self.device)
        yS = yS.to(self.device)
        score = self.forward(perturbed_x, random_timepoints, yS)

        # the loss weighting function may change, there are a few options that we will experiment on
        if self.speed_balanced:
            s = 2 / (torch.ones(self.ncat - 1, device= self.device) + torch.arange(self.ncat - 1, 0, -1,
                                                                                      device=self.device).float())
        else:
            s = torch.ones(self.ncat - 1, device= self.device)

        
        perturbed_v = sb._inverse(perturbed_x, prevent_nan=True).detach()
        
        loss = torch.mean(torch.mean(
            1 / (torch.sqrt(time_dependent_weights))[random_t][(...,) + (None,) * (x.ndim - 1)] * s[
                (None,) * (x.ndim - 1)] * perturbed_v * (1 - perturbed_v) * (
                        ddsm.gx_to_gv(score, perturbed_x, create_graph=True) - ddsm.gx_to_gv(perturbed_x_grad,
                                                                                    perturbed_x)) ** 2, dim=(1)))
        self.avg_loss += loss.item() * x.shape[0]
        self.num_items += x.shape[0]

        self.log("loss", loss)
        wandb.log({"loss": loss})
        return loss
    
    def on_train_batch_end(self, outputs, *args):
        self.log("average-loss", self.avg_loss / self.num_items)

        wandb.log({"Epoch average loss": self.avg_loss / self.num_items})
        wandb.log({"Epoch": self.current_epoch})
            
        return {"Epoch average loss": self.avg_loss / self.num_items} 

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr= self.lr)

