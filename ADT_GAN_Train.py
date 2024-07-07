import xarray as xr
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from gravity import Gravity as G
from torch import clamp
from pytorch_msssim import ssim
from torch.utils.data import DataLoader, TensorDataset, random_split
from coliolis_param import Coliolis_parameter as fparam
from model import *
from utils import *
from data_processing import *





directory='./Dataset/'
batch_size=32
input_dim = 200
starting_epoch=0

train_loader, validation_loader, temporal_info=create_train_and_validation_dataset(directory,batch_size)








device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
temporaly_dim=temporal_info.shape[1]

generator = Generator(input_dim, n_layers=4,temporaly_dim=temporaly_dim).to(device)
discriminator = Discriminator((1, 36, 86),temporaly_dim=temporaly_dim).to(device)



if torch.cuda.is_available():
  generator = nn.DataParallel(generator).cuda()
  discriminator = nn.DataParallel(discriminator).cuda()

generator.apply(weights_init)
discriminator.apply(weights_init)

# generator.apply(weights_init)
# discriminator.apply(weights_init)

# Loss and optimizers
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))





# # Training function with loss tracking
def train_gan(generator, discriminator, optimizer_g, optimizer_d, epochs, batch_size, train_loader, save_interval, lambda_gp=10):
  torch.autograd.set_detect_anomaly(True)
  # print(next(iter(train_loader)).shape)

  # Lists to store loss values
  d_losses = []
  g_losses = []
  EPOCHS=[]
  d_losses_mean = []
  g_losses_mean = []

  for epoch in range(starting_epoch,epochs):
    for i,batch in enumerate(train_loader):
      # print(batch[0].shape)
      for _ in range(5):  # Train discriminator more frequently
        discriminator.zero_grad()
        real_data = batch[0][:, :, :, :86].to(device).float()  # Ensure the data is of type float32
        batch_time=batch[0][:, 0, 0, 86:].to(device)
        batch_size_real = real_data.size(0)
        real_data = real_data.view(batch_size_real, 1, 36, 86).to(device)

        noise = torch.randn(batch_size_real, input_dim, device=device)
        fake_data = generator(noise,batch_time)

        critic_real = discriminator(real_data,batch_time).view(-1)
        critic_fake = discriminator(fake_data.detach(),batch_time).view(-1)
        gp = gradient_penalty(real_data, fake_data, discriminator,batch_time, lambda_gp)

        d_loss = -(torch.mean(critic_real) - torch.mean(critic_fake)) + gp
        d_loss.backward()
        optimizer_d.step()
        EPOCHS.append(epoch+i)

      generator.zero_grad()
      noise = torch.randn(batch_size_real, input_dim, device=device)
      fake_data = generator(noise,batch_time)
      g_loss_adv = -torch.mean(discriminator(fake_data, batch_time).view(-1))
      g_loss_ssim = ssim_loss(fake_data, real_data)
      g_loss = g_loss_adv + g_loss_ssim  # Combining adversarial and SSIM losses

      g_loss.backward()
      optimizer_g.step()

      # Store losses
      d_losses.append(d_loss.item())
      g_losses.append(g_loss.item())

    if (epoch+i) % save_interval == 0:
        save_image_and_models(fake_data,real_data, generator, discriminator, epoch)
        print(f'Epoch [{epoch}/{epochs}]  Loss_D: {d_loss.item()}, Loss_G: {g_loss.item()}')
    d_losses_mean.append(np.mean([dl for dl in d_losses]))
    g_losses_mean.append(np.mean([gl for gl in g_losses]))
    print(f'Epoch [{epoch}/{epochs}]  Mean Loss_D: {d_losses_mean[-1]}, Mean Loss_G: {g_losses_mean[-1]}')
  # Plot losses
  plt.figure(figsize=(10, 5))
  plt.plot(d_losses, label='Discriminator Loss')
  plt.plot(g_losses, label='Generator Loss')
  plt.xlabel('Iterations')
  plt.ylabel('Loss')
  plt.legend()
  plt.title('GAN Loss During Training')
  plt.savefig(f'./image_train/gen_disc_training_loss_epoch_{epoch}.png',dpi=300)
  plt.show()
  return d_losses,g_losses,EPOCHS,g_losses_mean,d_losses_mean





def save_image_and_models(generated_data, real_image, generator, discriminator, epoch):


  mean_generated_data = generated_data.mean(dim=0)
  mean_real_data = real_image.mean(dim=0)
  if mean_generated_data.ndim==2 :

    mean_generated_data = mean_generated_data.reshape(36,86)
    mean_real_data = mean_real_data.reshape(36,86)

  image =mean_generated_data
  image = image.detach().cpu().numpy().reshape(36, 86)
  mean_real_data=mean_real_data.cpu().numpy().reshape(36, 86)
  # print(image.shape)
  # Latitude and Longitude coordinates
  latitudes = np.linspace(29.5, 47, 36)
  longitudes = np.linspace(-5.5, 37, 86)
  lon, lat = np.meshgrid(longitudes, latitudes)
  g = G(lat)
  f = fparam(lat)
  dx = 6371e3 * np.cos(np.radians(lat)) * np.radians(0.5)
  dy = 6371e3 * np.radians(0.5)

  ugeo = -(g / f) * np.gradient(image, axis=0) / dy
  vgeo = (g / f) * np.gradient(image, axis=1) / dx

  ugeo_real= -(g / f) * np.gradient(mean_real_data, axis=0) / dy
  vgeo_real = (g / f) * np.gradient(mean_real_data, axis=1) / dx

  Wfake_test=np.sqrt(ugeo**2+vgeo**2)
  Wreal_test=np.sqrt(ugeo_real**2+vgeo_real**2)

  ugeo_real[Wreal_test>0.36]=0
  vgeo_real[Wreal_test>0.36]=0
  image[Wreal_test>0.36]= 0

  ugeo[Wfake_test>0.36]=0
  vgeo[Wfake_test>0.36]=0
  image[Wfake_test>0.36]=0

  # Plotting the image on the map
  fig, ax = plt.subplots(2, 2, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(14, 16))
  ax = ax.flatten()

  ax[0].coastlines(resolution='10m')
  ax[0].add_feature(cfeature.BORDERS)
  ax[0].add_feature(cfeature.LAND, facecolor='g', zorder=2)
  ax[0].add_feature(cfeature.RIVERS)
  ax[0].add_feature(cfeature.LAKES)

  # Plotting the data
  fig1 = ax[0].contourf(lon, lat, image, levels=10, transform=ccrs.PlateCarree(), cmap='turbo')
  fig.colorbar(fig1, ax=ax[0], ticks=fig1.levels, boundaries=fig1.levels, orientation='horizontal', label='Generated Value of SSH(in m)')

  ax[0].set_title(f"Generated Image of SSH at Epoch {epoch}")

  #-------------------------------------------------------------------------------------
  ax[1].set_title(f" Image of Real SSH observation at Epoch {epoch}")
  ax[1].coastlines(resolution='10m')
  ax[1].add_feature(cfeature.BORDERS)
  ax[1].add_feature(cfeature.LAND, facecolor='g', zorder=2)
  ax[1].add_feature(cfeature.RIVERS)
  ax[1].add_feature(cfeature.LAKES)

  # Plotting the data
  fig1 = ax[1].contourf(lon, lat, mean_real_data, levels=10, transform=ccrs.PlateCarree(), cmap='turbo')
  fig.colorbar(fig1, ax=ax[1], ticks=fig1.levels, boundaries=fig1.levels, orientation='horizontal', label='Real Value of SSH(in m)')



  #--------------------------------------------------------------------------

  ax[2].coastlines(resolution='10m')
  ax[2].add_feature(cfeature.BORDERS)
  ax[2].add_feature(cfeature.LAND, facecolor='g', zorder=2)
  ax[2].add_feature(cfeature.RIVERS)
  ax[2].add_feature(cfeature.LAKES)

  ucm = 100 * ugeo
  vcm = 100 * vgeo
  Wcm = np.sqrt(ucm**2 + vcm**2)
  bounds = np.linspace(0, int(np.nanmax(Wcm))+1, 10)
  scales = 300
  if np.nanmax(Wcm) > 40:
      scales = 850
      fig2 = ax[2].quiver(lon, lat, ucm, vcm, Wcm, transform=ccrs.PlateCarree(), cmap='turbo', scale=scales)
  else:
      scales = 350
      fig2 = ax[2].quiver(lon, lat, ucm, vcm, Wcm, transform=ccrs.PlateCarree(), cmap='turbo', scale=scales)

  fig.colorbar(fig2, ax=ax[2], ticks=bounds, boundaries=bounds, orientation='horizontal', label='Geostrophic current velocity (in cm/s) from SSH.')

  ax[2].set_title(f"Velocity from generated SSH at Epoch {epoch}")

  #------------------------------------------------------------------------------------------------

  ax[3].coastlines(resolution='10m')
  ax[3].add_feature(cfeature.BORDERS)
  ax[3].add_feature(cfeature.LAND, facecolor='g', zorder=2)
  ax[3].add_feature(cfeature.RIVERS)
  ax[3].add_feature(cfeature.LAKES)

  ucm = 100 * ugeo_real
  vcm = 100 * vgeo_real
  Wcm = np.sqrt(ucm**2 + vcm**2)
  bounds = np.linspace(0, int(np.nanmax(Wcm))+1, 10)
  scales = 300
  if np.nanmax(Wcm) > 40:
      scales = 850
      fig3 = ax[3].quiver(lon, lat, ucm, vcm, Wcm, transform=ccrs.PlateCarree(), cmap='turbo', scale=scales)
  else:
      scales = 350
      fig3 = ax[3].quiver(lon, lat, ucm, vcm, Wcm, transform=ccrs.PlateCarree(), cmap='turbo', scale=scales)

  fig.colorbar(fig3, ax=ax[3], ticks=bounds, boundaries=bounds, orientation='horizontal', label='Geostrophic current velocity (in cm/s) from SSH.')

  ax[3].set_title(f"Velocity from real SSH observation at Epoch {epoch}")


  plt.savefig(f"./image_training/gen_image_epoch_add_noise_{epoch}.png", dpi=350)
  plt.show()

#     plt.show()

#     # Save model weights
  torch.save(generator.state_dict(), f"./checkpoints/gen_weights_epoch_{epoch}.pth")
  torch.save(discriminator.state_dict(), f"./checkpoints/disc_weights_epoch_{epoch}.pth")

# # Example usage
d_losses,g_losses,EPOCHS,g_losses_mean,d_losses_mean=train_gan(generator, discriminator,  optimizer_g, optimizer_d, epochs=100, batch_size=batch_size, train_loader=train_loader, save_interval=5)
