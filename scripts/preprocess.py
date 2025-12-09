import os
from PIL import Image
import argparse

def rotate_imgs(image_dir):
    img_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(".png")])

    img_paths = [os.path.join(image_dir, f) for f in img_files]

    for i in range(3,8):
        img_pth = img_paths[i]

        img = Image.open(img_pth)
        img = img.transpose(Image.ROTATE_180)

        # print(f"Processing image at {img_pth}")

        img.save(img_pth)

def main():
    parser = argparse.ArgumentParser(description="Process two directories of background/source images in batch")
    parser.add_argument("--background_dir", type=str, help="directory of background images", default=r"resources/raw_data/empty_shot/FRAMES/")
    parser.add_argument("--source_dir", type=str, help="directory of source images", default=r"resources/raw_data/caroline_shot/FRAMES/")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame number (inclusive).")
    parser.add_argument("--end_frame", type=int, default=24, help="End frame number (exclusive).")

    args = parser.parse_args()

    for frame_idx in range(args.start_frame, args.end_frame):
        source_dir_path = os.path.join(args.source_dir, f"frame{frame_idx:04d}")
        rotate_imgs(source_dir_path)
        print(f"Processing frame {frame_idx} complete.")



    # rotate_imgs(args.background_dir)

if __name__ == '__main__':
    main()