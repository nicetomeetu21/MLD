
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from ddim_functions.denoising import ddpm_steps2
def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, y_0):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)

        loss = F.mse_loss(self.model(x_t, y_0, t), noise, reduction='none')
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, skip):
        super().__init__()

        self.model = model
        self.T = T//skip
        self.select_t = range(0, T, skip)
        # self.betas = 1-alphas
        self.register_buffer('betas',torch.linspace(beta_1, beta_T, T).double())



    def forward(self, x_T, y_0, y_1):
        """
        Algorithm 2.
        """
        x_0 = ddpm_steps2(x=x_T, seq = self.select_t, model = self.model, b = self.betas, y = y_0, y1 = y_1)
        return torch.clip(x_0, -1, 1)
        # res  =torch.cat(res, dim=0)
        # return res


