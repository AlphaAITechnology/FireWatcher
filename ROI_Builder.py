from libs.CreateROIMasks import get_roi_mask, draw_roi_mask
import argparse
import cv2 as cv
import numpy as np
import os


def circle_builder(img_path, save_path, size):
    
    if not (os.path.exists(img_path) and os.path.isfile(img_path)):
        raise ValueError("File Not Found")
    else:
        if (os.path.exists(save_path)):
            if os.path.isfile(save_path):
                os.remove(save_path)
            else:
                raise ValueError("Save Path not available")

        img = cv.imread(img_path)
        mask = get_roi_mask(img, size)
        np.savetxt(save_path, mask, delimiter=',')

def polygon_builder(img_path, save_path, draw_on_picture):
    
    if not (os.path.exists(img_path) and os.path.isfile(img_path)):
        raise ValueError("File Not Found")
    else:
        if (os.path.exists(save_path)):
            if os.path.isfile(save_path):
                os.remove(save_path)
            else:
                raise ValueError("Save Path not available")

        img = cv.imread(img_path)
        mask = draw_roi_mask(img, draw_on_picture)
        np.savetxt(save_path, mask, delimiter=',')

def main():
    parser = argparse.ArgumentParser(description='Draw Floor and Bin Simulation')
    parser.add_argument('--img_path', type=str, help='path to image', default=None)
    parser.add_argument('--save_path', type=str, help='path to mask directory', default=None)
    parser.add_argument('--size', type=int, help='radius of roi', default=30)
    parser.add_argument('--polygon', action=argparse.BooleanOptionalAction, default=False, help='draw a irregular polygon for ROI (deafult is circle)')
    parser.add_argument('--draw_on_picture', action=argparse.BooleanOptionalAction, default=False, help='draw directly on image (default is using canvas)')


    args = parser.parse_args()

    if not ((args.img_path is None) or (args.save_path is None)):
        try:
            if (args.polygon or args.draw_on_picture):
                polygon_builder(args.img_path, args.save_path, args.draw_on_picture)
            else:    
                circle_builder(args.img_path, args.save_path, args.size)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    main()