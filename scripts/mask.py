import argparse

import torch
import torchvision.io as io
import torchvision.transforms.functional as F

def compare(background_file: str, source_file: str) -> None:
    device = torch.device('cuda')
    precision = torch.float32

    model = torch.jit.load(r'resources/models/torchscript_resnet101_fp32.pth')
    model.backbone_scale = 0.25
    model.refine_mode = 'sampling'
    model.refine_sample_pixels = 80_000

    model = model.to(device)
    src = io.decode_image(io.read_file(source_file)).unsqueeze(0).to(device, dtype=precision) / 255.0
    bgr = io.decode_image(io.read_file(background_file)).unsqueeze(0).to(device, dtype=precision) / 255.0

    print(src.shape)
    print(bgr.shape)

    pha, _ = model(src, bgr)[:2]

    # Save pha and fgr to files
    F.to_pil_image(pha[0].cpu()).save("resources/mask.png")

def main():
    parser = argparse.ArgumentParser(description="Load two PNG images")
    parser.add_argument("--background", type=str, help="image file with only background environment", default=r"./resources/raw_data/empty_shot/FRAMES/frame0000/stream00_frame0000.png")
    parser.add_argument("--source", type=str, help="image file with subject within background", default=r"./resources/raw_data/caroline_shot/FRAMES/frame0000/stream00_frame0000.png")
    args = parser.parse_args()

    compare(args.background, args.source)

if __name__ == "__main__":
    main()
