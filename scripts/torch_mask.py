import argparse
import os
import torch
import torch.nn.functional as nnF
from torchvision.io import read_image
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

import numpy as np

from torchvision.models.segmentation import (
    fcn_resnet50, FCN_ResNet50_Weights,
    fcn_resnet101, FCN_ResNet101_Weights,
    deeplabv3_resnet50, DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet101, DeepLabV3_ResNet101_Weights,
    lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights,
)

plt.rcParams["savefig.bbox"] = "tight"

# ---------------------------------------
# Show in grid (kept from your version)
# ---------------------------------------
def show(imgs, out_path=None, title=None):
    if not isinstance(imgs, list):
        imgs = [imgs]

    cols = 4
    rows = (len(imgs) + cols - 1) // cols
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 4, rows * 4))
    fig.suptitle(title or "", fontsize=16)
    axs = axs.flatten() if isinstance(axs, (list, np.ndarray)) else [axs]

    for i, ax in enumerate(axs):
        if i < len(imgs):
            img = imgs[i].detach().cpu()
            ax.imshow(F.to_pil_image(img))
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, bbox_inches="tight", dpi=200)
        plt.close()
        print(f"✅ Saved grid: {out_path}")
    else:
        plt.show()


# ---------------------------------------
# Run segmentation models on a directory
# ---------------------------------------
def run_models_on_dir(image_dir, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)

    models = [
        (fcn_resnet50, FCN_ResNet50_Weights, "fcn_resnet50"),
        (fcn_resnet101, FCN_ResNet101_Weights, "fcn_resnet101"),
        (deeplabv3_resnet50, DeepLabV3_ResNet50_Weights, "deeplabv3_resnet50"),
        (deeplabv3_resnet101, DeepLabV3_ResNet101_Weights, "deeplabv3_resnet101"),
        (lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights, "lraspp_mobilenet_v3_large"),
    ]

    # Gather all .png images
    image_files = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(".png")
    ])
    if not image_files:
        print(f"No PNG images found in {image_dir}")
        return

    print(f"Found {len(image_files)} images.")

    for build_model, weights_cls, name in models:
        print(f"\n=== Running {name} ===")
        weights = weights_cls.DEFAULT
        model = build_model(weights=weights).to(device).eval()
        preprocess = weights.transforms()
        sem_classes = weights.meta["categories"]
        sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}

        # preprocess and batch
        imgs = [read_image(f) for f in image_files]
        batch = torch.stack([preprocess(img) for img in imgs]).to(device)

        with torch.no_grad():
            output = model(batch)["out"]
            probs = nnF.softmax(output, dim=1)

        if "person" not in sem_class_to_idx:
            print(f"{name} has no 'person' class; skipping.")
            continue

        person_idx = sem_class_to_idx["person"]
        person_masks = [probs[i, person_idx] for i in range(batch.shape[0])]

        out_path = os.path.join(output_dir, f"{name}.png")
        show(person_masks, out_path=out_path, title=name)


# ---------------------------------------
# Entrypoint
# ---------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Try multiple segmentation models and save grid plots.")
    parser.add_argument("--image_dir", type=str, help="Directory containing PNG images to segment.", default="./resources/raw_data/caroline_shot/FRAMES/frame0000/")
    parser.add_argument("--output_dir", type=str, default="./RESULTS/segmentation/", help="Directory to save outputs.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    run_models_on_dir(args.image_dir, args.output_dir, device)


if __name__ == "__main__":
    main()
