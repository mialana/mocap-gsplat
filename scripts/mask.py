import argparse
import os
import torch
import torchvision.io as io
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import numpy as np

# ------------------ BATCHED compare() ------------------
def compare(background_files, source_files):
    device = torch.device("cuda")
    precision = torch.float32

    # Load model once
    model = torch.jit.load(r"resources/models/torchscript_resnet101_fp32.pth")
    model.backbone_scale = 0.25
    model.refine_mode = "sampling"
    model.refine_sample_pixels = 80_000
    model = model.to(device)

    # Load and stack all images
    src_tensors, bgr_tensors = [], []
    for bgr_path, src_path in zip(background_files, source_files):
        src = io.decode_image(io.read_file(src_path)).to(device, dtype=precision) / 255.0
        bgr = io.decode_image(io.read_file(bgr_path)).to(device, dtype=precision) / 255.0
        src_tensors.append(src)
        bgr_tensors.append(bgr)

    # Stack into batched tensors
    src_batch = torch.stack(src_tensors, dim=0)
    bgr_batch = torch.stack(bgr_tensors, dim=0)
    print(f"Batched input: {src_batch.shape}")

    # Run model once on entire batch
    with torch.no_grad():
        pha, _ = model(src_batch, bgr_batch)[:2]

    # Return all alpha mattes to CPU
    return [pha[i].cpu() for i in range(pha.shape[0])]
# -------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Process two directories of background/source images in batch")
    parser.add_argument("--background_dir", type=str, help="directory of background images", default=r"resources/raw_data/empty_shot/FRAMES/frame0000")
    parser.add_argument("--source_dir", type=str, help="directory of source images", default=r"resources/raw_data/caroline_shot/FRAMES/frame0000")
    args = parser.parse_args()

    bg_dir = args.background_dir
    src_dir = args.source_dir

    # Match filenames present in both directories
    bg_files = sorted([f for f in os.listdir(bg_dir) if f.lower().endswith(".png")])
    src_files = sorted([f for f in os.listdir(src_dir) if f.lower().endswith(".png")])
    common_files = [f for f in bg_files if f in src_files]

    if not common_files:
        print("No matching filenames found between the two directories.")
        return

    print(f"Found {len(common_files)} matching image pairs.")

    # Build full paths
    bg_paths = [os.path.join(bg_dir, f) for f in common_files]
    src_paths = [os.path.join(src_dir, f) for f in common_files]

    # Batch process all
    masks = compare(bg_paths, src_paths)

    # ---------------- matplotlib grid identical to earlier ----------------
    cols = 4
    rows = (len(masks) + cols - 1) // cols
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 4, rows * 4))
    fig.suptitle("backgroundmattingv2_resnet101_fp32", fontsize=16)
    axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]

    for i, ax in enumerate(axs):
        if i < len(masks):
            img = masks[i]
            ax.imshow(F.to_pil_image(img), cmap="gray")
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()

    # Save output
    os.makedirs("RESULTS/segmentation", exist_ok=True)
    out_path = "RESULTS/segmentation/backgroundmattingv2_resnet101_fp32.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"✅ Saved grid: {out_path}")


if __name__ == "__main__":
    main()
