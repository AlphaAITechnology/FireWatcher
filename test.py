import cv2 as cv
import numpy as np
from libs.CreateROIMasks import get_roi_mask

img = cv.imread("./test.png")
mask = get_roi_mask(img)
mask = np.stack((mask, mask, mask), axis=2)


blue = np.zeros_like(mask)
blue[:,:,0] = 255
img = np.where(mask==1, mask, img)

cv.namedWindow("H", cv.WINDOW_NORMAL)
cv.imshow("H", img)
cv.waitKey(0)
cv.destroyAllWindows()