import numpy as np         
import torch             
import torch.nn as nn  
import torch.nn.functional as F  
from Network_layers import *






class Generator(nn.Module):
    def __init__(self, input_dim, n_layers,temporaly_dim):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            Mapping_Network(input_dim+temporaly_dim, n_layers),
            nn.Linear(input_dim+temporaly_dim, 128 * 9 * 21),  # Input: (N, input_dim) -> Output: (N, 128*9*21)
            nn.ReLU(True)
        )
        self.deconv = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 256, kernel_size=(4, 5), stride=(2, 2), padding=(1, 1), output_padding=(0, 1)),  # Output: (N, 256, 18, 42)
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            ResidualBlock(256),
            nn.Conv2d(256, 128, kernel_size=(4, 5), stride=(2, 2), padding=(1, 1)),  # Output: (N, 128, 18, 42)
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            ResidualBlock(128),
            SelfAttention(128),
            nn.ConvTranspose2d(128, 64, kernel_size=(4, 5), stride=(2, 2), padding=(1, 1), output_padding=(0, 1)),  # Output: (N, 64, 36, 86)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            ResidualBlock(64),
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 5), stride=(2, 2), padding=(1, 1), output_padding=(0, 1)),  # Output: (N, 32, 36, 86)
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            ResidualBlock(32),
            SelfAttention(32),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # Output: (N, 32, 36, 86)
            nn.Tanh()
        )
        self.last_layer = nn.Conv2d(32, 1, kernel_size=(1, 5), stride=(1, 1), padding=(0, 0))

    def forward(self, x,temporaly_data):
        x=torch.cat((x,temporaly_data),dim=1)
        x = self.fc(x)
        x = x.view(-1, 128, 9, 21)  # Reshape: (N, 128*9*21) -> (N, 128, 9, 21)
        x = self.deconv(x)  # Apply transpose convolutions
        x = self.last_layer(x)
        return x
    
    
    
    
    
    
class Discriminator(nn.Module):
    def __init__(self, input_shape,temporaly_dim):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),  # Input: (N, 1, 36, 86) -> Output: (N, 32, 18, 43)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            ResidualBlock(32),
            nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),  # Input: (N, 32, 18, 43) -> Output: (N, 64, 9, 22)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            ResidualBlock(64),
            SelfAttention(64),
            nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),  # Input: (N, 64, 9, 22) -> Output: (N, 128, 5, 12)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=(4, 4), stride=(1, 2), padding=(2, 2)),  # Input: (N, 128, 5, 12) -> Output: (N, 128, 6, 7)
            ResidualBlock(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 7))  # Ensure consistent output size (N, 128, 6, 7)
        self.vector=nn.Flatten()
        self.fc = nn.Sequential(
              # Input: (N, 128, 6, 7) -> Output: (N, 128*6*7)
            nn.Linear(128 * 6 * 7+temporaly_dim, 1),  # Input: (N, 128*6*7) -> Output: (N, 1)
            nn.Sigmoid()
        )

    def forward(self, x,temporaly_data):
        # noise=torch.randn(x.size())*0.1
        # noise=noise.to(device)
        # print(device)
        # print('all devices: ',x.device,noise.device)
        # noisy_input=x+noise
        x=add_noise(x)
        x = self.conv(x)
        x = self.adaptive_pool(x)
        x_vector=self.vector(x)
        x=torch.cat((x_vector,temporaly_data),dim=1)
        x = self.fc(x)
        return x
