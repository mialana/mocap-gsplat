import os
from PIL import Image
import argparse

def rotate_imgs(image_dir):
    img_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(".png")])

    img_paths = [os.path.join(image_dir, f) for f in img_files]

    for i in range(3,8):
        img_pth = img_paths[i]

        print(f"Processing image at {img_pth}")

        img = Image.open(img_pth)
        img = img.transpose(Image.ROTATE_180)

        img.save(img_pth)

def main():
    parser = argparse.ArgumentParser(description="Process two directories of background/source images in batch")
    parser.add_argument("--background_dir", type=str, help="directory of background images", default=r"resources/raw_data/empty_shot/FRAMES/frame0000")
    parser.add_argument("--source_dir", type=str, help="directory of source images", default=r"resources/raw_data/caroline_shot/FRAMES/frame0000")
    args = parser.parse_args()

    rotate_imgs(args.background_dir)
    rotate_imgs(args.source_dir)

if __name__ == '__main__':
    main()