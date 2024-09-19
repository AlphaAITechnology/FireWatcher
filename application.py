import argparse
import cv2 as cv
import datetime
import gzip
import json
import numpy as np
import os
import queue
import requests as req
import threading
import time
import torch


def ImageSending_IO():
    base_url = "https://fire-api.alphaaitech.com"
    x_api_token = None
    with open("./.env", 'r') as env_file:
        x_api_token = json.load(env_file)["x_api_token"]

    if x_api_token is None:
        elegant_shutdown.put(True)
    print("x_api_token loaded from .env file")

    while elegant_shutdown.empty():
        while(not sending_images_q.empty()):
            img_path = sending_images_q.get()
            camera_id = os.path.basename(img_path).split('@')[1].split('.')[0]

            with open(img_path, "rb") as files_:
                # storing file
                file_upload_response = req.post(
                    f"{base_url}/file",
                    files={'file': (img_path, files_, 'image/webp')},
                    headers={"x-api-token": x_api_token},
                )
            if (file_upload_response.status_code == 201):
                upload_response = json.loads(file_upload_response.text)
                alert_response = req.post(
                        f"{base_url}/alert-record",
                        headers={"x-api-token": x_api_token},
                        data={
                            "url": upload_response["fileUrl"],
                            "type": "PERSON",
                            "cameraId": camera_id
                        }
                    )
                
                if (alert_response.status_code >= 200 or alert_response.status_code <= 203):
                    print(alert_response.text)
                else:
                    print("Alert Upload Unsucessful:\t", alert_response.status_code)
            else:
                print("File Upload Unsucessful:\t", file_upload_response.status_code)

            # Delete image from disks
            os.remove(img_path)
            del img_path

    elegant_shutdown.put(True)



def ImageSaving_IO():
    if os.path.exists("./saved_images"):
        if not os.path.isdir("./saved_images"):
            os.remove("./saved_images")
            os.mkdir("./saved_images")
    else:
        os.mkdir("./saved_images")
    
    while elegant_shutdown.empty():
        while(not printing_images_q.empty()):
            camera_TID, img = printing_images_q.get()
            img_path = f"./saved_images/{camera_TID}.webp"
            cv.imwrite(img_path, img)
            sending_images_q.put(img_path)
            
            del img
            del camera_TID

    elegant_shutdown.put(True)


def detect(results, conf, classes):
    res = results.pandas().xyxy[0]
    res = res[res["confidence"] >= conf]
    res[res["class"].isin(classes)]
    return res if res.size>0 else None

def ImageAnalysis():
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")
    with gzip.open("./FloorMask.csv.gz") as mask_gz:
        roi_mask = np.loadtxt(mask_gz, delimiter=',').astype(np.uint8)
        roi_mask = np.stack((roi_mask, roi_mask, roi_mask), axis=2)

    minimum_confidence = 0.4


    while elegant_shutdown.empty():
        while not capture_images_q.empty():
            camera_TID, img = capture_images_q.get()
            results = model(img)
            results = detect(results, minimum_confidence, [0])

            if results is not None:
                results_np = results[["xmin", "ymin", "xmax", "ymax"]].to_numpy().tolist()
                analysis_image = np.zeros_like(roi_mask)
                for x_min, y_min, x_max, y_max in results_np:
                    analysis_image[int(y_min):int(y_max), int(x_min):int(x_max)] = 1
                
                roi_intersect = np.where((roi_mask * analysis_image)==1, 1, 0).astype(np.uint64)
                if np.add.reduce(roi_intersect.reshape((-1,)))>0:
                    printing_images_q.put((camera_TID, img))
                

            del img
            del camera_TID

    elegant_shutdown.put(True)


def ImageCapture_IO():
    cameras_link = None
    cameras_id = None

    while cameras_link is None:
        if (not cameras_links.empty()):
            cameras_il = cameras_links.get()
            cameras_link = cameras_il["link"]
            cameras_id = cameras_il["uid"]

    cap = cv.VideoCapture(cameras_link, cv.CAP_FFMPEG)
    fpso = cap.get(cv.CAP_PROP_FPS) * 2
    count = -1       # counting number of frames read
    frame_const = fpso//2 # reading every fifth frame

    while(elegant_shutdown.empty()):
        try:
            while cap.isOpened():
                count += 1
                ret = cap.grab()
                if (count%frame_const == 0 and ret):
                    ret, frame = cap.retrieve()
                    if ret and capture_images_q.empty():
                        print(f"Image Retrieved and Sent:\t{count}")
                        capture_images_q.put((f"{datetime.datetime.now().isoformat()}@{cameras_id}", frame))

                time.sleep(1/fpso)
        except Exception as e:
            cap.release()
            elegant_shutdown.put(True)
            print(e)

    elegant_shutdown.put(True)




def main():

    parser = argparse.ArgumentParser(description='Watch Cameras for Humans')
    parser.add_argument('--rtsp', type=str, help='rtsp link for camera', default=None)
    parser.add_argument('--uuid', type=str, help='rtsp link for camera', default=None)

    parser.add_argument('--env_camera', type=int, help='index of camera from .env.json file', default=0)
    
    args = parser.parse_args()

    cameras = None
    if (args.rtsp is None):
        with open("./.env.json", 'r') as env_file:
            jdata = json.load(env_file)
            cameras = jdata["cameras"] if "cameras" in jdata else []
        cameras_links.put(cameras[args.env_camera])
    if (not ((args.rtsp is None) or (args.uuid is None))):
        cameras_links.put({
                "uid": args.uuid,
                "link": args.rtsp
            })
    

    
    

    
    p1 = threading.Thread(target=ImageCapture_IO)
    p2 = threading.Thread(target=ImageAnalysis)
    p3 = threading.Thread(target=ImageSaving_IO)
    p4 = threading.Thread(target=ImageSending_IO)

    p1.start()
    p2.start()
    p3.start()
    p4.start()



cameras_links = queue.Queue()
capture_images_q = queue.Queue()
printing_images_q = queue.Queue()
sending_images_q = queue.Queue()
elegant_shutdown = queue.Queue()


if __name__ == "__main__":
    main()
    # elegant_shutdown.put(True)