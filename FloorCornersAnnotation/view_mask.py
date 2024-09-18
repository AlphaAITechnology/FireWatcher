import numpy as np
import cv2 as cv
import gzip
import argparse
import os

def view_mask(mask_path, img_path, delimiter, mask_flip):
    assert (os.path.exists(mask_path) and os.path.exists(img_path)) and (os.path.isfile(mask_path) and os.path.isfile(img_path)), "Mask and Image file not found"

    with gzip.open(mask_path, 'rt') as mask_file_buffer:
        mask = np.loadtxt(mask_file_buffer, delimiter=delimiter).astype(np.uint8)
        mask = np.ones_like(mask)-mask if mask_flip else mask
        mask = np.stack((mask, mask, mask), axis=2)
    
    img = cv.imread(img_path)
    cv.imshow("Masked Image", img * mask)
    cv.waitKey(0)
    cv.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='View Mask on Images')
    parser.add_argument('--img_path', type=str, help='path to image file', default=None)
    parser.add_argument('--mask_path', type=str, help='path to mask file', default=None)
    parser.add_argument('--mask_flip', type=bool, help='invert mask binary', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--delimiter', type=str, help='Choice of csv delimiter', default=',')

    args = parser.parse_args()
    view_mask(args.mask_path, args.img_path, args.delimiter, args.mask_flip)

if __name__ == "__main__":
    main()

