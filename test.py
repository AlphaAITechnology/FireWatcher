import cv2 as cv
import numpy as np
from libs.create_floor_hotspot import create_labels

floor_image = cv.imread("./test.png")
# blue_square = np.zeros((1000,1000), dtype=np.uint8)
# blue_square[450:550, 450:550] = 1
# blue_square = np.stack((blue_square*255, blue_square, blue_square), axis=2)

blue_square = np.zeros((100,100, 3), dtype=np.uint8)
blue_square[:,:,0] = 255


floor_corners, bins_positions = create_labels(floor_image, (0,0,0))
# M_grid2Image, status = cv.findHomography(np.array([[450,450], [450,550], [550,550], [550,550]]), floor_corners)
M_grid2Image, status = cv.findHomography(np.array([[0,0], [0,100], [100,100], [100,0]]), floor_corners)
M_Image2grid, status = cv.findHomography(floor_corners, np.array([[0,0], [100, 0], [100,100], [0, 100]]))

# im_out = cv.warpPerspective(blue_square, M_grid2Image, (floor_image.shape[1], floor_image.shape[0]))
# im_out = cv.warpPerspective(floor_image, M_Image2grid, (blue_square.shape[1], blue_square.shape[0]))

floor_corners_transled = cv.perspectiveTransform(np.expand_dims(floor_corners.astype(np.float32), axis=0), M_Image2grid)
bins_positions_transled = cv.perspectiveTransform(np.expand_dims(bins_positions.astype(np.float32), axis=0), M_Image2grid)
print(bins_positions_transled.astype(np.uint8))

# floor_map = np.zeros_like(blue_square)
# for x, y in floor_corners_transled.reshape((-1,2)).tolist():
#     floor_map = cv.circle(floor_map, (int(x),int(y)), 10, (0,255,0), -1)

# print(floor_corners)
# print(floor_corners_transled)

# cv.namedWindow("l", cv.WINDOW_NORMAL)
# cv.imshow("l", im_out)
# cv.waitKey(0)
# cv.destroyAllWindows()

