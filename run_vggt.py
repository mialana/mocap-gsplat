import torch
from external.vggt.vggt.models.vggt import VGGT
from external.vggt.vggt.utils.load_fn import load_and_preprocess_images

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

print(f"Device: {device}")
print(f"Data Type: {dtype}")

print("Downloading model if running for the first time...")

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

print("1. Model initialized successfully")

print("Loading and preprocessing images...")
# Load and preprocess example images
image_names = ["colmap_project_dir/images/stream00_frame0320.png", "colmap_project_dir/images/stream01_frame0320.png", "colmap_project_dir/images/stream02_frame0320.png", "colmap_project_dir/images/stream03_frame0320.png", "colmap_project_dir/images/stream04_frame0320.png", "colmap_project_dir/images/stream05_frame0320.png", "colmap_project_dir/images/stream06_frame0320.png", "colmap_project_dir/images/stream07_frame0320.png"]
images = load_and_preprocess_images(image_names).to(device)
print(f"2. Preprocess done")

print("Predicting attributes...")
with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps.
        predictions = model(images)

print("3. Prediction done")