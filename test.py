import cv2 as cv
import numpy as np
from libs.create_floor_hotspot import create_labels

floor_image = cv.imread("./test.png")
blue_square = np.ones((100,100), dtype=np.uint8)
blue_square = np.stack((blue_square*255, blue_square, blue_square), axis=2)


floor_corners, bins_positions = create_labels(floor_image, (0,0,0))
M, status = cv.findHomography(np.array([[0,0], [0,100], [100,100], [100,0]]), floor_corners)
im_out = cv.warpPerspective(blue_square, M, (floor_image.shape[1], floor_image.shape[0]))


cv.namedWindow("l", cv.WINDOW_NORMAL)
cv.imshow("l", im_out)
cv.waitKey(0)
cv.destroyAllWindows()


print(M)