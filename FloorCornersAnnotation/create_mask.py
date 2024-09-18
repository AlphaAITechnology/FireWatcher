import numpy as np
import cv2 as cv
import argparse
import os
# import gzip
# import time




def create_labels(save_dir, save_fname, mask_color, img_dir, img_path=None, mask_flip=False):
    mask_color_ = [min([max([int(i), 0]), 255]) for i in mask_color.split(",")]

    if (img_path is not None):
        img = None
        points = [[]]
        try:
            if (os.path.exists(img_path) and os.path.isfile(img_path)):
                img = cv.imread(img_path)
            else:
                raise ValueError("File not Found")
        except Exception as e:
            print(e)
            exit()
            
        def draw_circle(event,x,y,flags,param):
            if event == cv.EVENT_LBUTTONDBLCLK:
                cv.circle(img,(x,y),5,(255,0,0),-1)
                points[-1].append((x,y))


        cv.namedWindow("Image", cv.WINDOW_NORMAL)
        cv.setMouseCallback("Image", draw_circle)
        while True:
            img_c = img.copy()
            for point in points:
                img_c = cv.fillPoly(img_c, [cv.convexHull(np.array(point), returnPoints=True).reshape((-1,2))], mask_color_) if len(point)>2 else img_c
            
            cv.imshow("Image", img_c)
            if cv.waitKey(1) & 0xFF == ord("q"):
                if len(points[-1]) == 0:
                    break
                points.append([])
        cv.destroyAllWindows()

        # print(cv.convexHull(np.array(points), returnPoints=True).reshape((-1,2)))         
        mask = np.zeros_like(img)
        for point in points:
            if len(point)>2:
                mask = cv.fillPoly(mask, [cv.convexHull(np.array(point), returnPoints=True).reshape((-1,2))], [1,1,1])
        mask = np.ones_like(mask)-mask if not mask_flip else mask

        try:
            if (os.path.exists(save_dir) and os.path.isdir(save_dir)):
                save_path = os.path.join(save_dir, "mask.csv" if save_fname is None else save_fname)
                if (os.path.exists(save_path)):
                    raise ValueError("Mask already exists; will be overwritten")
                else:
                    np.savetxt(save_path, mask[:,:,0], delimiter=',')
            else:
                raise ValueError("Save Path not found")
        except Exception as e:
            print(e)
            exit()

        
    elif (img_dir is not None):
        # handle directory of images
        pass


def main():
    parser = argparse.ArgumentParser(description='Draw Annotations on Images')
    parser.add_argument('--img_dir', type=str, help='path to image directory', default=None)
    parser.add_argument('--img_name', type=str, help='path to image', default=None)
    parser.add_argument('--save_dir', type=str, help='path to mask directory', default=None)
    parser.add_argument('--save_fname', type=str, help='path to mask directory', default=None)
    parser.add_argument('--mask_color', type=str, help='three digits delimited by comma \"255,255,255\"', default="120,120,120")
    parser.add_argument('--mask_flip', type=bool, help='invert mask binary', action=argparse.BooleanOptionalAction, default=False)
    
    args = parser.parse_args()

    if (((args.img_dir is None) and (args.img_name is None)) or (args.save_dir is None)):
        print("Path of Images and Masks must be given")
    else:
        if (args.img_name is None):
            create_labels(args.save_dir, args.save_fname, args.mask_color, args.img_dir, None, args.mask_flip)
        else:
            create_labels(args.save_dir, args.save_fname, args.mask_color, None, args.img_name, args.mask_flip)



if __name__ == "__main__":
    main()

