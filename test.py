import cv2 as cv
import numpy as np
from libs.create_floor_hotspot import create_labels



# def get_roi_mask(floor_img):
    
    

floor_image = cv.imread("./test.png")
floor_simulation = np.zeros((100,100, 3), dtype=np.uint8)

## Read values from GUI input
floor_corners, bins_positions = create_labels(floor_image, (0,0,0))

## Calculate Homography matrices for simulation calculation and inverse conversion
M_grid2Image, status = cv.findHomography(np.array([[0,0], [100, 0], [100,100], [0, 100]]), floor_corners)
M_Image2grid, status = cv.findHomography(floor_corners, np.array([[0,0], [100, 0], [100,100], [0, 100]]))

## Get position of garbage bins across a simulated floor
floor_corners_transled = cv.perspectiveTransform(np.expand_dims(floor_corners.astype(np.float32), axis=0), M_Image2grid)
bins_positions_transled = cv.perspectiveTransform(np.expand_dims(np.array(bins_positions).reshape((-1,2)).astype(np.float32), axis=0), M_Image2grid)


# # Redefine floor_simulation incase of curvature inaccuracies
# coors_ = np.squeeze(bins_positions_transled.astype(np.uint8), axis=0).tolist() + np.squeeze(floor_corners_transled.astype(np.uint8), axis=0).tolist() + [[floor_simulation.shape[1], floor_simulation.shape[0]]]
# x_max, y_max = max([x for x,_ in coors_]), max([y for _, y in coors_])
# floor_simulation = np.zeros((x_max, y_max, 3), dtype=np.uint8)

## Dray circle for ROI detetctions
for x,y in np.squeeze(bins_positions_transled.astype(np.uint8), axis=0).tolist():
    # Can adjust radius here
    floor_simulation = cv.circle(floor_simulation, (x,y), 30, (0,255,0), -1)

## Draw mask out to real world perspective
im_out = cv.warpPerspective(floor_simulation, M_grid2Image, (floor_image.shape[1], floor_image.shape[0]))

## Draw on to image for visualizaztion
im_out_m = np.where(im_out[:,:,1] == 0, 1, 0).astype(np.uint8)
im_out_m = np.stack((im_out_m, im_out_m, im_out_m), axis=2)
im_out = (floor_image * im_out_m) + im_out

## Display
cv.namedWindow("l", cv.WINDOW_NORMAL)
cv.imshow("l", im_out)
cv.waitKey(0)
cv.destroyAllWindows()



