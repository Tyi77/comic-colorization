import os
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr

# PSNR calculation function
def calculate_psnr(real_images, generated_images):
    """
    Calculate the average PSNR between real and generated images.
    Assumes that images in both lists correspond one-to-one.

    Parameters:
        real_images (list of PIL.Image): List of real images.
        generated_images (list of PIL.Image): List of generated images.

    Returns:
        float: Average PSNR score across all image pairs.
    """
    if len(real_images) != len(generated_images):
        raise ValueError("The number of real and generated images must be the same.")

    psnr_values = []
    for real, generated in zip(real_images, generated_images):
        real_np = np.array(real)
        generated_resized = generated.resize(real.size, Image.BICUBIC)
        generated_np = np.array(generated_resized)

        # Ensure both images have the same shape
        if real_np.shape != generated_np.shape:
            raise ValueError("Image shapes do not match after resizing: {} vs {}".format(real_np.shape, generated_np.shape))

        # Calculate PSNR
        psnr_value = psnr(real_np, generated_np, data_range=255)
        psnr_values.append(psnr_value)

    return np.mean(psnr_values)

# Load images from a folder
def load_images_from_folder(folder_path):
    images = []
    filenames = []
    for filename in sorted(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        try:
            image = Image.open(img_path).convert('RGB')
            images.append(image)
            filenames.append(filename)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue
    return images, filenames

# Paths to your real and generated images
real_folder = 'real'  # Folder where real images are stored
generated_folder = 'output'  # Folder where generated images are stored

# Load the images
real_images, real_filenames = load_images_from_folder(real_folder)
generated_images, gen_filenames = load_images_from_folder(generated_folder)

# Ensure file names match for comparison
# if real_filenames != gen_filenames:
#     raise ValueError("The filenames in the real and generated folders do not match. Ensure corresponding images have the same filenames.")

# Calculate PSNR score
psnr_score = calculate_psnr(real_images, generated_images)
print(f"Average PSNR Score: {psnr_score:.2f} dB")