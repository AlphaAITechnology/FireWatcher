import cv2 as cv
import numpy as np
import argparse
import gzip


def draw_mask(img_path, mask_path, mask_color):
    mask_color = [int(i) for i in mask_color.split(',')]
    img = cv.imread(img_path)

    with gzip.open(mask_path) as mask_buffer:
        mask = np.loadtxt(mask_buffer, delimiter=',', dtype=np.uint8)
        mask = np.stack((mask, mask, mask), axis=2)


    cv.namedWindow("H", cv.WINDOW_NORMAL)
    cv.imshow("H", np.where(mask == 1, mask*mask_color, img).astype(np.uint8))
    cv.waitKey(0)
    cv.destroyAllWindows()


def main():
    # python visualize_floor_mask.py --img_path ./FloorSampleImage.png --mask_path FloorMask.csv.gz --mask_color 0,0,25

    parser = argparse.ArgumentParser(description='Draw Floor and Bin Simulation')
    parser.add_argument('--img_path', type=str, help='path to image', default=None)
    parser.add_argument('--mask_path', type=str, help='path to mask', default=None)
    parser.add_argument('--mask_color', type=str, help='colour of mask', default="0,255,0")


    args = parser.parse_args()

    if not (args.img_path is None or args.mask_path is None or args.mask_color is None):
        draw_mask(args.img_path, args.mask_path, args.mask_color)
    


if __name__ == "__main__":
    main()



