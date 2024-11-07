from PIL import Image
import os
from concurrent.futures import ProcessPoolExecutor

# Directory containing the images
input_dir = '../ffhq-dataset/images1024x1024'
output_dir = '../ffhq-dataset/images224x224'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to resize a single image and save it in the same folder structure
def resize_image(filepath):
    # Calculate the relative path from the input directory
    rel_path = os.path.relpath(filepath, input_dir)
    # Calculate the destination path in the output directory
    output_path = os.path.join(output_dir, rel_path)
    
    # Ensure the subdirectory exists in the output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Open, resize, and save the image
    with Image.open(filepath) as img:
        img_resized = img.resize((224, 224), Image.LANCZOS)
        img_resized.save(output_path)
    
    return filepath  # Return filepath for tracking

# Gather all PNG images from input directory and subdirectories
image_files = []
for root, _, files in os.walk(input_dir):
    for file in files:
        if file.endswith('.png'):
            image_files.append(os.path.join(root, file))

# Process images in parallel
with ProcessPoolExecutor() as executor:
    results = list(executor.map(resize_image, image_files))

print(f"Resized {len(results)} images successfully!")
