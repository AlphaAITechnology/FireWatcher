import numpy as np
import cv2 as cv
import torch
import queue
import threading
# import gzip
import time
# import datetime
import json

def ImageSending_IO():
    pass
def ImageSaving_IO():
    pass
def ImageAnalysis():
    # model = torch.hub.load("ultralytics/yolov5", "yolov5s")
    cv.namedWindow("Feed", cv.WINDOW_NORMAL)

    while elegant_shutdown.empty():
        while(not capture_images_q.empty()):
            cv.imshow("Feed", capture_images_q.get())
        cv.waitKey(1)
    cv.destroyAllWindows()
    elegant_shutdown.put(True)


def ImageCapture_IO():
    while(elegant_shutdown.empty()):
        if (not cameras_links.empty()):
            cameras_ = cameras_links.get()
        cap = cv.VideoCapture(cameras_, cv.CAP_FFMPEG)
        fpso = cap.get(cv.CAP_PROP_FPS) * 2
        count = 0       # counting number of frames read
        frame_const = 5 # reading every fifth frame

        try:
            while cap.isOpened():
                read = cap.grab()
                count += 1
                if (count%frame_const == 0 and read):
                    ret, frame = cap.retrieve()
                    if ret:
                        capture_images_q.put(frame)
                time.sleep(1/fpso)
        except Exception as e:
            cap.release()
            print(e)
        finally:
            elegant_shutdown.put(True)
    elegant_shutdown.put(True)


def main():
    cameras = None
    with open("./.env.json", 'r') as env_file:
        jdata = json.load(env_file)
        cameras = jdata["cameras"] if "cameras" in jdata else []
    
    for camera in cameras[:1]: # for now only read one camera
        cameras_links.put(camera)

    
    p1 = threading.Thread(target=ImageCapture_IO)
    p2 = threading.Thread(target=ImageAnalysis)

    p1.start()
    p2.start()



cameras_links = queue.Queue()
capture_images_q = queue.Queue()
elegant_shutdown = queue.Queue()


if __name__ == "__main__":
    main()