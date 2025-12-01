import sys
import os

vggt_dir = os.path.abspath('external/vggt')
sys.path.insert(0, vggt_dir)

import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

print(f"Device: {device}")
print(f"Data Type: {dtype}")

print("Downloading model if running for the first time...")

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT.from_pretrained("facebook/VGGT-1B", cache_dir="/home/aliu/.cache/vggt").to(device)

print("Model initialized successfully")

image_directory = "resources/colmap_project_dir/images"

image_names = []
for root, _, files in os.walk(image_directory):
    for file in files:
        image_names.append(os.path.join(root, file))

print(f'Processing images in "{image_directory}"')

# Load and preprocess example images (replace with your own image paths)
images = load_and_preprocess_images(image_names).to(device)

with torch.no_grad():
    with torch.amp.autocast("cuda", dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps.
        predictions = model(images)