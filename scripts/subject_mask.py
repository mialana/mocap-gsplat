import cv2
import numpy as np
import argparse

def compare(background_file: str, source_file: str) -> None:
    background = cv2.imread(background_file)
    # background = cv2.cvtColor(background, cv2.COLOR_BGR2LAB)
    
    source = cv2.imread(source_file)
    # source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)

    diff = cv2.absdiff(source, background)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    radial_mask = np.zeros(source.shape[:2], np.uint8)
    center = (source.shape[1] // 2, source.shape[0] // 2)
    radius = min(center) - 200
    cv2.circle(radial_mask, center, radius, 255, -1)

    gray_diff = cv2.bitwise_and(gray_diff, gray_diff, mask=radial_mask)

    _, mask = cv2.threshold(gray_diff, 15, 255, cv2.THRESH_BINARY)

    kernel = np.ones((10, 10),np.uint8)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # Remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # Fill small gaps

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    person_mask = np.uint8(labels == largest_label) * 255

    cv2.imshow("preview", mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()




def main():
    parser = argparse.ArgumentParser(description="Load two PNG images")
    parser.add_argument("background", type=str, help="image file with only background environment")
    parser.add_argument("source", type=str, help="image file with subject within background")
    args = parser.parse_args()

    compare(args.background, args.source)

if __name__ == "__main__":
    main()
