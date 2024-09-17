import numpy as np
import cv2 as cv
import torch
import queue
import threading
# import gzip
# import time
# import datetime
import json

def ImageSending_IO():
    pass
def ImageSaving_IO():
    pass
def ImageAnalysis():
    # model = torch.hub.load("ultralytics/yolov5", "yolov5s")
    while True:
        pass
def ImageCapture_IO():
    pass


def main():
    cameras = None
    with open("./.env.json", 'r') as env_file:
        jdata = json.load(env_file)
        cameras = jdata["cameras"] if "cameras" in jdata else []
    
    for camera in cameras:
        pass




# capture_images_q = queue.Queue()
# elegant_shutdown = queue.Queue()


if __name__ == "__main__":
    main()