import cv2 as cv
import numpy as np

img = cv.imread("./FloorSampleImage.png")
mask = np.loadtxt("./FloorMask.csv", delimiter=',', dtype=np.uint8)
mask = np.stack((mask, mask, mask), axis=2)


green = np.zeros_like(mask)
green[:,:,1] = 255

cv.namedWindow("H", cv.WINDOW_NORMAL)
cv.imshow("H", np.where(mask==1, green, img))
cv.waitKey(0)
cv.destroyAllWindows()