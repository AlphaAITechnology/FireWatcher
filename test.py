import cv2 as cv
import numpy as np


cap = cv.VideoCapture("rtsp://admin:12345678a@121.202.153.80:554")

cv.namedWindow("H", cv.WINDOW_NORMAL)
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        cv.imshow("H", frame)

cv.waitKey(0)
cv.destroyAllWindows()
cap.release()