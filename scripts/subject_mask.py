import cv2
import numpy as np
import argparse

def compare(background_file: str, subject_file: str) -> None:
    pass


def main():
    parser = argparse.ArgumentParser(description="Load two PNG images")
    parser.add_argument("background", type="str", help="image file with only background environment")
    parser.add_argument("subject", type="str", help="image file with subject within background")
    args = parser.parse_args()

if __name__ == "__main__":
    main()
