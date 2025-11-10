import argparse
import os

import torch

from torchvision.io import read_image
import torchvision.transforms.functional as F
import torch.nn.functional as nnF

from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights

import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda')

weights = FCN_ResNet50_Weights.DEFAULT
model = fcn_resnet50(weights=weights).to(device).eval()
preprocess = weights.transforms()

sem_classes = weights.meta["categories"]
sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}

plt.rcParams["savefig.bbox"] = "tight"

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    cols = 4
    rows = (len(imgs) + cols - 1) // cols
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 4, rows * 4))
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < len(imgs):
                img = imgs[i * cols + j]
                img = img.detach().cpu()
                axs[i, j].imshow(F.to_pil_image(img))
                axs[i, j].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


def compare(image_files: str) -> None:
    imgs = [read_image(f) for f in image_files]
    batch = torch.stack([preprocess(img) for img in imgs]).to(device)
   
    with torch.no_grad():
        output = model(batch)["out"]
        probs = nnF.softmax(output, dim=1)

    person_idx = sem_class_to_idx["person"]

    person_masks = [probs[i, person_idx] for i in range(batch.shape[0])]

    show(person_masks)

def main():
    parser = argparse.ArgumentParser(description="Load two PNG images")
    parser.add_argument("--image_dir", type=str, help="directory containing all images to process", default="./resources/raw_data/caroline_shot/FRAMES/frame0000/")
    args = parser.parse_args()

    image_files = []

    for entry in os.listdir(args.image_dir):
        full_path = os.path.join(args.image_dir, entry)
        if os.path.isfile(full_path) and full_path.endswith(".png"):
            image_files.append(full_path)

    compare(image_files)

if __name__ == "__main__":
    main()
