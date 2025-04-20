import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, transform=None, data_dir=None, is_test=False):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.data_dir = data_dir or ("test_data_v2" if is_test else "train_data")
        self.is_test = is_test
        
        # Ensure the file_name column exists
        if 'file_name' not in self.data.columns and not self.is_test:
            # If first column has no name, rename it
            if self.data.columns[0] == 'Unnamed: 0' or self.data.columns[0].isdigit():
                self.data = self.data.rename(columns={self.data.columns[0]: 'file_name'})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the image filename using the column name if available, otherwise use index
        if 'file_name' in self.data.columns:
            img_name = str(self.data['file_name'][idx])
        else:
            # For test data that might have a different format
            img_name = str(self.data.iloc[idx, 0])
        
        # Remove 'train_data/' prefix if it exists in the filename
        if img_name.startswith('train_data/'):
            img_name = img_name.replace('train_data/', '', 1)

        if img_name.startswith('test_data_v2/'):
            img_name = img_name.replace('test_data_v2/', '', 1)
            
        # Construct the full image path
        img_path = os.path.join(self.data_dir, img_name)
        
        # Handle potential missing files gracefully
        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, IOError) as e:
            print(f"Error loading image {img_path}: {e}")
            # Create a black image as placeholder
            image = Image.new('RGB', (224, 224), color='black')
        
        # Get label correctly
        label = -1
        if 'label' in self.data.columns:
            label = int(self.data['label'][idx])
        elif not self.is_test and len(self.data.columns) > 1:
            label = int(self.data.iloc[idx, 1])
            
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_loaders(train_csv, batch_size=32, img_size=224, data_dir='train_data'):
    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Create dataset
    dataset = CustomImageDataset(csv_file=train_csv, transform=transform, data_dir=data_dir)
    
    # Read the CSV file to get labels for stratification
    df = pd.read_csv(train_csv)
    labels = df['label'].values if 'label' in df.columns else df.iloc[:, 1].values
    
    # Create stratified train/val split (80/20)
    train_idx, val_idx = train_test_split(
        np.arange(len(dataset)),
        test_size=0.2,
        shuffle=True,
        stratify=labels,
        random_state=42
    )
    
    # Create Subset datasets
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    
    # Verify stratification
    train_labels = [labels[i] for i in train_idx]
    val_labels = [labels[i] for i in val_idx]
    print(f"Train set: {len(train_dataset)} images, Class 0: {train_labels.count(0)}, Class 1: {train_labels.count(1)}")
    print(f"Val set: {len(val_dataset)} images, Class 0: {val_labels.count(0)}, Class 1: {val_labels.count(1)}")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    return train_loader, val_loader

def get_test_loader(test_csv, batch_size=32, img_size=224, data_dir=''):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    test_data = CustomImageDataset(
        csv_file=test_csv, 
        transform=transform, 
        data_dir=data_dir,
        is_test=True
    )
    
    test_loader = DataLoader(
        test_data, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    return test_loader