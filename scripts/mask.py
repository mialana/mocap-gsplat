import argparse
import os
import torch
import torchvision.io as io
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda")
precision = torch.float32

model = torch.jit.load(r"resources/models/torchscript_resnet101_fp32.pth")
model.backbone_scale = 0.25
model.refine_mode = "sampling"
model.refine_sample_pixels = 80000
model.refine_threshold = 0.1
model = model.to(device)

def show(masks):
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
    out_path = f"RESULTS/segmentation/backgroundmattingv2_{str(model.backbone_scale).replace('.', '_')}_{model.refine_mode}_{str(model.refine_threshold).replace('.', '_')}.png"
    out_path = f"RESULTS/segmentation/backgroundmattingv2_{str(model.backbone_scale).replace('.', '_')}_{model.refine_mode}_{model.refine_sample_pixels}.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved grid: {out_path}")


# ------------------ BATCHED compare() ------------------
def compare(background_files, source_files):

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
        pha, fgr = model(src_batch, bgr_batch)[:2]

    com = pha * fgr

    # Return all alpha mattes to CPU
    return pha, com
# -------------------------------------------------------------

def save(img_stack, common_files, output_dir="", ):
    os.makedirs(output_dir, exist_ok=True)
    for idx, img_data in enumerate(img_stack):
        img = F.to_pil_image(img_data)
        img.save(os.path.join(output_dir, common_files[idx]))

def main():
    parser = argparse.ArgumentParser(description="Process two directories of background/source images in batch")
    parser.add_argument("--background_dir", type=str, help="directory of background images", default=r"resources/raw_data/empty_shot/")
    parser.add_argument("--source_dir", type=str, help="directory of source images", default=r"resources/raw_data/caroline_shot/")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame number (inclusive).")
    parser.add_argument("--end_frame", type=int, default=24, help="End frame number (exclusive).")
    args = parser.parse_args()

    for frame_idx in range(args.start_frame, args.end_frame):
        source_dir_path = os.path.join(args.source_dir, "FRAMES", f"frame{frame_idx:04d}")
        background_dir_path = os.path.join(args.background_dir, "FRAMES", f"frame{frame_idx:04d}")

        bg_files = sorted([f for f in os.listdir(background_dir_path) if f.lower().endswith(".png")])
        src_files = sorted([f for f in os.listdir(source_dir_path) if f.lower().endswith(".png")])
        common_files = [f for f in bg_files if f in src_files]

        # Match filenames present in both directories

        if not common_files:
            print("No matching filenames found between the two directories.")
            return

        print(f"Found {len(common_files)} matching image pairs.")

        # Build full paths
        bg_paths = [os.path.join(background_dir_path, f) for f in common_files]
        src_paths = [os.path.join(source_dir_path, f) for f in common_files]

        # Batch process all
        masks, composite = compare(bg_paths, src_paths)

        output_dir_path = os.path.join(args.source_dir, "PROCESSED", f"frame{frame_idx:04d}")

        save(composite, common_files, output_dir_path)

        print(f"Processing frame {frame_idx} complete.")

    # show(masks)

    


if __name__ == "__main__":
    main()
