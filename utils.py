

import numpy as np         
import torch             
import torch.nn as nn 
from pytorch_msssim import ssim



device= 'cuda' if torch.cuda.is_available() else 'cpu'

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.orthogonal_(m.weight,gain=torch.nn.init.calculate_gain('relu'))
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
            
            

def gradient_penalty(real_data, fake_data, discriminator,time_data, lambda_gp=10):
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)  # Random weight between real and fake data
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates = interpolates.clone().detach().requires_grad_(True)  # Now it's a leaf node
    outputs = discriminator(interpolates,time_data)
    gradients = torch.autograd.grad(
        outputs, interpolates, grad_outputs=torch.ones_like(outputs), create_graph=True
    )[0]
    gradients = gradients.view(batch_size, -1)  # Flatten
    return lambda_gp * ((gradients.norm(2, dim=1) - 1)**2).mean()

def ssim_loss(fake, real):
    return 1 - ssim(fake, real, data_range=1, size_average=True)