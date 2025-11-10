import argparse

import torch
from torchvision_

from torchvision.models.segmentation import fcn_resnet50

model = fcn_resnet50(pretrained=True, progress=True)
model = model.eval()

def compare(background_file: str, source_file: str) -> None:
    device = torch.device('cuda')
    precision = torch.float16

    model = torch.jit.load('PATH_TO_MODEL.pth')
    model.backbone_scale = 0.25
    model.refine_mode = 'sampling'
    model.refine_sample_pixels = 80_000

    model = model.to(device)

    src = io.decode_image(io.read_file(source_file))
    bgr = io.decode_image(io.read_file(background_file))

    pha, fgr = model(src, bgr)[:2]



def main():
    parser = argparse.ArgumentParser(description="Load two PNG images")
    parser.add_argument("background", type=str, help="image file with only background environment")
    parser.add_argument("source", type=str, help="image file with subject within background")
    args = parser.parse_args()

    compare(args.background, args.source)

if __name__ == "__main__":
    main()
