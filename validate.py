from collections import OrderedDict
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score,mean_absolute_error
from gravity import Gravity as G
from coliolis_param import Coliolis_parameter as fparam
from model import *
from data_processing import *
# from ADT_GAN_Train import generator  
import torch

batch_size=32
input_dim = 200  # Example input dimension for the generator
num_samples = 224
nlayers=4


directory='./Dataset/'
path='./checkpoints/gen_weights_epoch_36.pth'
states=torch.load(path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Strip the 'module.' prefix

train_loader, validation_loader, temporal_info = create_train_and_validation_dataset(directory,batch_size)
temporaly_dim=temporal_info.shape[1]




new_state_dict = OrderedDict()
for k, v in states.items():
    name = k[7:] if k.startswith('module.') else k  # Remove `module.` if it exists
    new_state_dict[name] = v
    
# Load the generator model
generator = Generator(input_dim=input_dim, layers=nlayers,temporaly_dim=temporaly_dim).to(device)
generator.load_state_dict(new_state_dict)#,map_location=torch.device('cpu')
generator.eval() 










# Assuming generator and discriminator are already defined and trained

# Function to generate and visualize samples
def generate_samples(generator, num_samples, input_dim, device,validation_set):
  All_generated_data=[]
  generator.eval()
  with torch.no_grad():
        # Generate random noise

        # Generate fake data

    for i,batch in enumerate(validation_set):

      # print('The shape of the batch is: ',batch[0].shape)
      real_data = batch[0][:, :, :, :86].to(device).float()  # Ensure the data is of type float32
      batch_time=batch[0][:, 0, 0, 86:].to(device)
      noise = torch.randn(batch_time.shape[0], input_dim, device=device)
      print('the noise and time data shape is: ',noise.shape,batch_time.shape)
      x=torch.cat((noise,batch_time),dim=1)
      fake_data = generator(noise,batch_time)
      # print(fake_data.shape)
      # print(real_data.shape)
      All_generated_data.append((fake_data,real_data))

  return All_generated_data

# # Function to visualize samples
# def visualize_samples(samples, num_samples=8, title="Generated Samples"):
#     fig, axes = plt.subplots(1, num_samples, figsize=(15, 15))
#     for i in range(num_samples):
#         axes[i].imshow(samples[i][0].cpu().numpy().transpose(1, 2, 0))
#         axes[i].axis('off')
#     plt.suptitle(title)
#     plt.show()

def visualize_ssh_and_velocity(generated_data, epoch):
    generated_image, real_image = generated_data

    mean_generated_data = generated_image.mean(dim=0)
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
    ax[1].set_title(f"Validation Image of SSH at Epoch {epoch}")
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

    ax[3].set_title(f"Velocity from Real SSH at Epoch {epoch}")


    plt.savefig(f"./image_validation/validated_image_epoch_complex_{epoch}.png", dpi=350)
    plt.show()

# Function to run validation loop
def validate_gan(generator, val_loader, input_dim, device, num_samples=8):
    generator.eval()  # Set generator to evaluation mode

    all_generated_samples = []

    with torch.no_grad():
      generated_samples = generate_samples(generator, 32, input_dim, device,val_loader)
      for i, data in enumerate(generated_samples):
            # Generate samples

        all_generated_samples.append(data[0])
            # Visualize some samples
            # if i == 0:  # Only visualize first batch
        visualize_ssh_and_velocity(data, i)

    # Concatenate all generated samples
    all_generated_samples = torch.cat(all_generated_samples)

    # Calculate metrics (e.g., FID, IS) if applicable
    # For simplicity, let's assume we are just visualizing

    return all_generated_samples


def generate_samples(generator, num_samples, input_dim, device,validation_set):
  All_generated_data=[]
  generator.eval()
  with torch.no_grad():
        # Generate random noise

        # Generate fake data

    for i,batch in enumerate(validation_set):

      # print('The shape of the batch is: ',batch[0].shape)
      real_data = batch[0][:, :, :, :86].to(device).float()  # Ensure the data is of type float32
      batch_time=batch[0][:, 0, 0, 86:].to(device)
      noise = torch.randn(batch_time.shape[0], input_dim, device=device)
      print('the noise and time data shape is: ',noise.shape,batch_time.shape)
      x=torch.cat((noise,batch_time),dim=1)
      fake_data = generator(noise,batch_time)
      # print(fake_data.shape)
      # print(real_data.shape)
      All_generated_data.append((fake_data,real_data))

  return All_generated_data




# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


generated_samples = validate_gan(generator, validation_loader, input_dim, device, num_samples)

All_generated_data=generate_samples(generator, num_samples, input_dim, device,validation_loader)


Testing_data=[]
Generated_data=[]
for i in range(len(All_generated_data)):
    xi,xj = All_generated_data[i]
    Generated_data.append(xi)
    Testing_data.append(xj)
    
    
final_testing=torch.cat(Testing_data,dim=0)
final_generated=torch.cat(Generated_data,dim=0)


# Flatten the images and create dummy target values
X = final_generated.detach().cpu().numpy().reshape(num_samples, -1)
y = final_testing.detach().cpu().numpy().reshape(num_samples, -1)  # Dummy target values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple machine learning model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2}")

# Compute per-pixel (per-feature) MSE and MAE
mse_per_pixel = mean_squared_error(y_test, y_pred, multioutput='raw_values')
mae_per_pixel = mean_absolute_error(y_test, y_pred, multioutput='raw_values')

# Aggregate metrics
mean_mse_per_pixel = np.mean(mse_per_pixel)
mean_mae_per_pixel = np.mean(mae_per_pixel)

print(f"Mean MSE per pixel: {mean_mse_per_pixel}")
print(f"Mean MAE per pixel: {mean_mae_per_pixel}")

plt.figure(figsize=[18,8])
plt.plot(mse_per_pixel,label='MSE per pixel')
plt.xlabel('Pixels Number')
plt.ylabel('MSE')
plt.legend()
plt.savefig('./loss_images/validation_loss_mse_COMPLEX.png',dpi=300)
plt.show()