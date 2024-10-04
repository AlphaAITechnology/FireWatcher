from ultralytics import YOLO
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



def ImageSending_IO():
    base_url = "https://fire-api.alphaaitech.com"
    x_api_token = None
    with open("./.env", 'r') as env_file:
        x_api_token = json.load(env_file)["x_api_token"]

    if x_api_token is None:
        elegant_shutdown.put(True)
    print("x_api_token loaded from .env file")

    while elegant_shutdown.empty():
        # Handle Fire Detection 
        while(not sending_images_f.empty()):
            img_path = sending_images_f.get()
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
                            "type": "FIRE",
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
            os.remove(img_path) ## --> TODO: Exists for debugging
            del img_path

        # Handle Human Detection 
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
            os.remove(img_path) ## --> TODO: Exists for debugging
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
        while(not printing_images_f.empty()):
            camera_TID, img = printing_images_f.get()
            img_path = f"./saved_images/f_{camera_TID}.webp"
            cv.imwrite(img_path, img)
            sending_images_f.put(img_path)
            
            del img
            del camera_TID

        while(not printing_images_q.empty()):
            camera_TID, img = printing_images_q.get()
            img_path = f"./saved_images/{camera_TID}.webp"
            cv.imwrite(img_path, img)
            sending_images_q.put(img_path)
            
            del img
            del camera_TID

    elegant_shutdown.put(True)


# def detect(results, conf, classes):
#     res = results.pandas().xyxy[0]
#     res = res[res["confidence"] >= conf]
#     res[res["class"].isin(classes)]
#     return res if res.size>0 else None



def FireAnalysis():

    model = YOLO("Weights/fire_v8l.pt") #v8l_100
    print("Fire Model Loaded")
    minimum_confidence = 0.4
    dec_window_size=6
    dec_window_approv=4
    dec_window_list_results=[]

    while elegant_shutdown.empty():
        try:
            while not capture_images_f.empty():
                camera_TID, img = capture_images_f.get()
                
                results = model(img, stream=True, conf=minimum_confidence, classes=[0,1]) # all classes for fire
                results = [result.boxes.xyxy.cpu().numpy() for result in results][0]
                
                
                dec_window_list_results.append(results) ## list of numpy # store image and model results 
                while (len(dec_window_list_results)>dec_window_size): # only stores static number of images
                    dec_window_list_results.pop(0) # remove oldest image


                if sum([(1 if res.shape[0]>0 else 0) for res in dec_window_list_results]) >= dec_window_approv: # we have 15+ out of 20 positives
                    printing_images_f.put((camera_TID, img))
                    dec_window_list_results.clear()
                
                del img
                del camera_TID
        except Exception as e:
            print(e)
            elegant_shutdown.put(True)
    elegant_shutdown.put(True)


def HumanAnalysis():
    model = YOLO("Weights/yolov8l.pt") 
    print("Human Model Loaded")
    with gzip.open("./FloorMask.csv.gz") as mask_gz:
        roi_mask = np.loadtxt(mask_gz, delimiter=',').astype(np.uint64)

    minimum_confidence = 0.4
    dec_window_size=6
    dec_window_approv=4
    dec_window_list_imgresults=[]

    while elegant_shutdown.empty():
        try:
            while not capture_images_q.empty():
                camera_TID, img = capture_images_q.get()
                results = results = model(img, stream=True, conf=minimum_confidence, classes=[0]) # only person class
                results = [result.boxes.xyxy.cpu().numpy() for result in results]

                

                dec_window_list_imgresults.append((img, results))
                while(len(dec_window_list_imgresults)>dec_window_size):
                    dec_window_list_imgresults.pop(0)
                

                if sum([(1 if sum([(1 if r[0].shape[0]>0 else 0) if len(r)>0 else 0 for r in res])>0 else 0) for _, res in dec_window_list_imgresults])>=dec_window_approv: # we have 15+ out of 20 positives
                    overlap_deg = [max([np.add.reduce(roi_mask[max([int(y2)-1, 0]), int(x1):int(x2)].reshape((-1,))) for x1,_,x2,y2 in res_[0].tolist()] if (len(res_)>0 and res_[0].shape[0]>0) else [0]) for _, res_ in dec_window_list_imgresults]
                    
                    best_img_idx, best_img_val = None, None
                    for bi, bv in enumerate(overlap_deg):
                        if ((best_img_val is None) or (best_img_idx is None)):
                            best_img_val = bv
                            best_img_idx = bi
                            continue

                        if (bv > best_img_val):
                            best_img_val = bv
                            best_img_idx = bi
                    
                    if best_img_val > 0: # make sure that best overlap is actually overlapping because max([0,0]) == 0
                        imgr = dec_window_list_imgresults[best_img_idx][0]
                        printing_images_q.put((camera_TID, imgr))
                        dec_window_list_imgresults.clear()
                        
                del img
                del camera_TID
        except Exception as e:
            print(e)
            elegant_shutdown.put(True)
    elegant_shutdown.put(True)






def ImageCapture_IO():
    cameras_link = None
    cameras_id = None

    while cameras_link is None:
        if (not cameras_links.empty()):
            cameras_il = cameras_links.get()
            cameras_link = cameras_il["link"]
            cameras_id = cameras_il["uid"]

    cap = cv.VideoCapture(cameras_link)
    fpso = cap.get(cv.CAP_PROP_FPS) * 2
    count = -1       # counting number of frames read
    frame_const = fpso//2 # reading every fifth frame

    recover = 0
    while(elegant_shutdown.empty()):
        try:
            while cap.isOpened():
                count += 1
                ret = cap.grab()

                if (not ret):
                    recover += 1
                    if (recover < 10):
                        print("Grab Failure")
                        cap.release()
                        cap = cv.VideoCapture(cameras_link)
                        continue
                    else:
                        raise ValueError("Grab Failure")

                if (count%(frame_const//5) == 0): # keeping it to 5 frames per second or less
                    ret, frame = cap.retrieve()
    
                    if (not ret):
                        raise ValueError("Retrieve Failure")

                    if ret and (capture_images_q.empty() and capture_images_f.empty()):
                        recover = 0
                        print(f"Sent Successful:\t{count}")
                        capture_images_q.put((f"{datetime.datetime.now().isoformat()}@{cameras_id}", frame[:,:,:]))
                        capture_images_f.put((f"{datetime.datetime.now().isoformat()}@{cameras_id}", frame[:,:,:]))
                        del frame



                time.sleep(1/fpso)
        except Exception as e:
            print(e)
        finally:
            cap.release()
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
    p2 = threading.Thread(target=HumanAnalysis)
    p3 = threading.Thread(target=ImageSaving_IO)
    p4 = threading.Thread(target=ImageSending_IO)
    p5 = threading.Thread(target=FireAnalysis)

    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()



cameras_links = queue.Queue()
capture_images_q = queue.Queue()
printing_images_q = queue.Queue()

capture_images_f = queue.Queue()
printing_images_f = queue.Queue()

sending_images_q = queue.Queue()
sending_images_f = queue.Queue()
elegant_shutdown = queue.Queue()


if __name__ == "__main__":
    main()
    # elegant_shutdown.put(True)
