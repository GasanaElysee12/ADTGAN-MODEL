import torch               
import numpy as np  
import xarray as xr
import os 
from torch.utils.data import DataLoader, TensorDataset, random_split         
from torchvision import transforms









def preprocess_temporal_info(dates):
    years = []
    months = []
    days = []
    hours = []
    minutes = []
    seconds = []

    for date in dates:
        year, month, day = date
        years.append(year)
        months.append(month - 1)  # zero-based for one-hot encoding
        days.append(day)


    # Normalize year
    years = np.array(years)
    if years.max() == years.min():
        years = np.zeros_like(years)  # Set to zero if all years are the same
    else:
        years = (years - years.min()) / (years.max() - years.min())

    # One-hot encode months
    months = np.eye(12)[months]  # 12 months

    # Normalize day, hour, minute, second
    days = np.array(days) / 31.0
#     hours = np.array(hours) / 23.0
#     minutes = np.array(minutes) / 59.0
#     seconds = np.array(seconds) / 59.0
# , hours[:, None], minutes[:, None], seconds[:, None]
    # Combine all temporal features
    temporal_info = np.concatenate([years[:, None], months, days[:, None]], axis=1)
    return torch.tensor(temporal_info, dtype=torch.float32)


def normalize_data(directory):
    
    files=[file for file in os.listdir(directory) if file.endswith('.nc')]
    print('The file name is: ',files[0])
    # Load and preprocess data
    data_link=os.path.join(directory,files[0])
    print('the dataset link: ',data_link)
    data = xr.open_dataset(data_link)
    ssh_data = data.adt.data
    ssh_data[np.isnan(ssh_data)] = 0.0
    ssh_data = (ssh_data - np.min(ssh_data)) / (np.max(ssh_data) - np.min(ssh_data))
    ssh_data = ssh_data.reshape((ssh_data.shape[0], 1, 36, 86))  # Shape (N, 1, 36, 86)

    # Convert data to tensor and ensure type is float32
    ssh_data = torch.tensor(ssh_data, dtype=torch.float32).clone().detach()
    
    Time_info=list(zip(data.time.dt.year.data.tolist(),\
                   data.time.dt.month.data.tolist(),\
                   data.time.dt.day.data.tolist()))
    temporal_info = preprocess_temporal_info(Time_info)
    
    return ssh_data,temporal_info



def data_transformer(data):

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
    ])
    
    return data_transforms(data)


def get_all_data_transformed(data):
    
    All_data=[]

    for image_data in data.numpy():
    # print(image_data.shape)
        d_trans=data_transformer(image_data.reshape((36, 86,1)))
        
        All_data.append(d_trans)
    
    transformed_data=torch.tensor([i.numpy().tolist() for i in All_data],dtype=torch.float32)
    
    return transformed_data




def create_train_and_validation_dataset(directory,batch_size):
    ssh_data, temporal_info = normalize_data(directory)
    transformed_data=get_all_data_transformed(ssh_data)
    
    
    tensor1_reshaped = temporal_info.view(7247, 1, 1, 14).expand(-1, -1, 36, -1)

    tensor1_reshape_together = torch.cat((tensor1_reshaped,tensor1_reshaped),dim=0)
    data_tensor_with_transform = torch.cat((ssh_data,transformed_data),dim=0)

    tensor1_reshape_together=tensor1_reshaped.clone()
    # data_tensor = torch.cat((data_tensor_with_transform,tensor1_reshape_together),dim=-1)
    data_tensor_with_transform=ssh_data.clone()
    data_tensor = torch.cat((data_tensor_with_transform,tensor1_reshape_together),dim=-1)
    # print(data_tensor.shape)
    # Create a TensorDataset
    dataset = TensorDataset(data_tensor)


    # Define the lengths of the datasets
    train_size = int(0.982 * len(dataset))
    validation_size = int(0.018 * len(dataset))

    validation_size=224
    train_size=len(dataset)-validation_size
    # Split the dataset
    train_dataset, validation_dataset= random_split(dataset, [train_size, validation_size])

    # Create DataLoaders

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, validation_loader, temporal_info
    
    
    
    
    
    
    
    
    
    
    
    
    
    

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
