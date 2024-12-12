import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import Inception_V3_Weights
from scipy.linalg import sqrtm
from PIL import Image

# FID calculation function
def calculate_fid(real_images, generated_images, batch_size=50, dims=2048, device='cuda'):
    # Load the Inception V3 model with weights parameter
    inception_model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1).to(device)
    inception_model.eval()
    
    def get_features(images):
        # Transform the images to match Inception V3 input requirements
        preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),  # Ensures images are exactly 299x299
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        images = torch.stack([preprocess(image).to(device) for image in images])
        
        with torch.no_grad():
            features = inception_model(images)
            features = features.view(features.size(0), -1)
        
        return features.cpu().numpy()
    
    # Get features for real images and generated images
    real_features = []
    for i in range(0, len(real_images), batch_size):
        real_batch = real_images[i:i+batch_size]
        real_features.append(get_features(real_batch))
    
    generated_features = []
    for i in range(0, len(generated_images), batch_size):
        generated_batch = generated_images[i:i+batch_size]
        generated_features.append(get_features(generated_batch))
    
    real_features = np.concatenate(real_features, axis=0)
    generated_features = np.concatenate(generated_features, axis=0)
    
    # Calculate the FID score
    mu_real = np.mean(real_features, axis=0)
    mu_gen = np.mean(generated_features, axis=0)
    
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_gen = np.cov(generated_features, rowvar=False)
    
    diff = mu_real - mu_gen
    covmean = sqrtm(sigma_real.dot(sigma_gen))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid_score = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2 * covmean)
    
    return fid_score

# Load images from a folder
def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        try:
            image = Image.open(img_path).convert('RGB')
            images.append(image)
        except:
            continue  # Skip if there's an error loading the image
    return images

# Paths to your real and generated images
real_folder = 'real'  # Folder where real images are stored
generated_folder = 'output'  # Folder where generated images are stored

# Load the images
real_images = load_images_from_folder(real_folder)
generated_images = load_images_from_folder(generated_folder)

# Calculate FID score
fid_score = calculate_fid(real_images, generated_images)
print(f"FID Score: {fid_score}")
