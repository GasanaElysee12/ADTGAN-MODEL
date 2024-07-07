import numpy as np         
import torch             
import torch.nn as nn  
import torch.nn.functional as F        


device= 'cuda' if torch.cuda.is_available() else 'cpu'

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.in1 = nn.InstanceNorm2d(in_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.in2 = nn.InstanceNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.in2(out)
        out += residual
        return out
    
    
    
    
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out
    
    
    
class Mapping_Network(nn.Module):
    def __init__(self, in_channels, n_layers):
        super(Mapping_Network, self).__init__()
        self.model = nn.Sequential()
        for i in range(n_layers - 1):
            self.model.append(nn.Linear(in_channels, in_channels))
            self.model.append(nn.ReLU(inplace=True))
        self.model.append(nn.Linear(in_channels, in_channels))

    def forward(self, x):
        return self.model(x)
    
    
def add_noise(inputs, mean=0, std=0.1):
  noise = torch.sin(torch.randn_like(inputs) * std + mean)
  noise = noise.to(device)
  return inputs + noise