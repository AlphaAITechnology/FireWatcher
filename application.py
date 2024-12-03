from ultralytics import YOLO
import argparse
import cv2 as cv
import datetime
import pytz
import gzip
import json
import numpy as np
import os
import queue
import requests as req
import threading
import time
import logging



def ImageSending_IO():
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='./LOGS/FILES/ImageSending.logs', encoding='utf-8', level=logging.INFO)


    base_url = "https://fire-api.alphaaitech.com"
    x_api_token = None
    with open("./.env", 'r') as env_file:
        x_api_token = json.load(env_file)["x_api_token"]

    if x_api_token is None:
        elegant_shutdown.put(True)
    print("x_api_token loaded from .env file")

    while elegant_shutdown.empty():
        dtm_ = datetime.datetime.now(pytz.utc).isoformat().split('+')[0]

        # Handle Fire Detection 
        while(not sending_images_f.empty()):
            img_path = sending_images_f.get()
            camera_tstamp, camera_id = os.path.basename(img_path).split('@')
            camera_tstamp = camera_tstamp.split('_')[1]
            camera_id = camera_id.split('.')[0]
            
            logger.info(f"FIRE:\tUploading Image:\t{img_path}")

            with open(img_path, "rb") as files_:
                # storing file
                file_upload_response = req.post(
                    f"{base_url}/file",
                    files={'file': (img_path, files_, 'image/webp')},
                    headers={"x-api-token": x_api_token},
                )
            
            logger.info(f"FIRE:\tUploading Image Response:\t{file_upload_response.status_code},\t{json.dumps(file_upload_response)}")

            if (file_upload_response.status_code == 201):
                upload_response = json.loads(file_upload_response.text)
                logger.info(f"FIRE:\tUploading URL:\t{upload_response["fileUrl"]}")
                alert_response = req.post(
                        f"{base_url}/alert-record",
                        headers={"x-api-token": x_api_token},
                        data={
                            "url": upload_response["fileUrl"],
                            "type": "FIRE",
                            "cameraId": camera_id,
                            "alertAt": camera_tstamp
                        }
                    )
                logger.info(f"FIRE:\tUploading URL Response:\t{alert_response.status_code},\t{json.dumps(alert_response)}")


                if (not (alert_response.status_code >= 200 or alert_response.status_code <= 203)):
                    logger.error(f"FIRE:\tURL Upload Unsuccessful_{dtm_}; response:{alert_response.status_code}")
            else:
                logger.error(f"FIRE:\tImage Upload Unsuccessful_{dtm_}; response:{file_upload_response.status_code}")

            # Delete image from disks
            os.remove(img_path) ## --> TODO: Exists for debugging
            del img_path

        # Handle Human Detection 
        while(not sending_images_q.empty()):
            img_path = sending_images_q.get()
            camera_tstamp, camera_id = os.path.basename(img_path).split('@')
            camera_id = camera_id.split('.')[0]


            logger.info(f"HUMAN:\tUploading Image:\t{img_path}")
            with open(img_path, "rb") as files_:
                # storing file
                file_upload_response = req.post(
                    f"{base_url}/file",
                    files={'file': (img_path, files_, 'image/webp')},
                    headers={"x-api-token": x_api_token},
                )
            logger.info(f"HUMAN:\tUploading Image Response:\t{file_upload_response.status_code},\t{json.dumps(file_upload_response)}")

            if (file_upload_response.status_code == 201):
                upload_response = json.loads(file_upload_response.text)
                logger.info(f"HUMAN:\tUploading URL:\t{upload_response["fileUrl"]}")
                alert_response = req.post(
                        f"{base_url}/alert-record",
                        headers={"x-api-token": x_api_token},
                        data={
                            "url": upload_response["fileUrl"],
                            "type": "PERSON",
                            "cameraId": camera_id,
                            "alertAt": camera_tstamp
                        }
                    )
                logger.info(f"HUMAN:\tUploading URL Response:\t{alert_response.status_code},\t{json.dumps(alert_response)}")
                
                if (not (alert_response.status_code >= 200 or alert_response.status_code <= 203)):
                    logger.error(f"HUMAN:\tURL Upload Unsuccessful_{dtm_}; response:{alert_response.status_code}")
            else:
                logger.error(f"HUMAN:\tImage Upload Unsuccessful_{dtm_}; response:{file_upload_response.status_code}")

            # Delete image from disks
            os.remove(img_path) ## --> TODO: Exists for debugging
            del img_path

    elegant_shutdown.put(True)



def ImageSaving_IO():
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='./LOGS/FILES/ImageSaving.logs', encoding='utf-8', level=logging.INFO)

    if os.path.exists("./saved_images"):
        if not os.path.isdir("./saved_images"):
            os.remove("./saved_images")
            os.mkdir("./saved_images")
    else:
        os.mkdir("./saved_images")
    
    while elegant_shutdown.empty():
        dtm_ = datetime.datetime.now(pytz.utc).isoformat().split('+')[0]
        while(not printing_images_f.empty()):
            camera_TID, img = printing_images_f.get()
            img_path = f"./saved_images/f_{camera_TID}.webp"
            cv.imwrite(img_path, img)
            sending_images_f.put(img_path)

            logger.info(f"Saved Fire Image:{dtm_}")
            
            del img
            del camera_TID

        while(not printing_images_q.empty()):
            camera_TID, img = printing_images_q.get()
            img_path = f"./saved_images/{camera_TID}.webp"
            cv.imwrite(img_path, img)
            sending_images_q.put(img_path)

            logger.info(f"Saved Human Image:{dtm_}")
            
            del img
            del camera_TID

    elegant_shutdown.put(True)



def FireAnalysis():
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='./LOGS/FILES/FireAnalysis.logs', encoding='utf-8', level=logging.INFO)

    model = YOLO("Weights/fire_v8l.pt")
    print("Fire Model Loaded")
    minimum_confidence = 0.55
    dec_window_size=25
    dec_window_approv=22
    dec_window_release= max(0, min(10, dec_window_approv//2))

    dec_window_list_results=[]
    seen_before = False

    while elegant_shutdown.empty():
        try:
            while not capture_images_f.empty():
                dtm_ = datetime.datetime.now(pytz.utc).isoformat().split('+')[0]
                camera_TID, img = capture_images_f.get()
                
                results = model(img, stream=True, conf=minimum_confidence, classes=[0], device='cuda:1', verbose=False) # all classes for fire 0: smoke, 1: fire
                results = [np.floor(result.boxes.xyxy.cpu().numpy()) for result in results][0]
                
                resulting_flag = 1 if results.shape[0]>0 else 0
                dec_window_list_results.append(resulting_flag) ## list of int flags; storing results

                if (resulting_flag==1):
                    logger.info(f"Fire Detected:\t{dtm_}")

                while (len(dec_window_list_results)>dec_window_size): # only stores static number of images
                    dec_window_list_results.pop(0) # remove oldest image


                if (not seen_before) and (sum(dec_window_list_results) >= dec_window_approv): # we have 5+ out of 50 positives
                    printing_images_f.put((camera_TID, img))
                    seen_before = True
                    logger.info("Fire Alert sent; deactivating alert")
                elif (seen_before) and (sum(dec_window_list_results) <= dec_window_release):
                    seen_before = False
                    logger.info("Fire Extinguished; deactivating alert")



                
                del img
                del camera_TID
        except Exception as e:
            print(e)
            elegant_shutdown.put(True)
    elegant_shutdown.put(True)


def HumanAnalysis():
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='./LOGS/FILES/HumanAnalysis.logs', encoding='utf-8', level=logging.INFO)

    model = YOLO("Weights/yolov8l.pt") 
    print("Human Model Loaded")
    with gzip.open("./FloorMask.csv.gz") as mask_gz:
        roi_mask = np.loadtxt(mask_gz, delimiter=',').astype(np.uint64)

    minimum_confidence = 0.6
    dec_window_size=15
    dec_window_approv=8 # Must be greater than zero
    dec_window_release= max(0, min(5, dec_window_approv//2))

    dec_window_list_imgresults=[]
    has_seen = False

    annotation_counter = 0

    while elegant_shutdown.empty():
        try:
            while not capture_images_q.empty():
                camera_TID, img = capture_images_q.get()
                results = model(img, stream=True, conf=minimum_confidence, classes=[0], device='cuda:1', verbose=False) # only person class
                results = [np.floor(result.boxes.xyxy.cpu().numpy()) for result in results] # bring to xyxy numpy

                #! Save image with label
                save_annotations = [result.tolist() for result in results if result.shape[0]>0]
                if (len(save_annotations)>0): # human detetcted; annotations to save exists
                    logger.info("Human Detected; Saving Frames to LOGS/HUMAN/")
                    with open(f"LOGS/HUMAN/{annotation_counter:0>8}.txt", 'w') as lf:
                        for coors in save_annotations:
                            lf.write(json.dumps(coors))
                    cv.imwrite(f"LOGS/HUMAN/{annotation_counter:0>8}.webp", img)
                    annotation_counter+=1
                #! End Saving Image with label
                
                # get max roi intersection of each detection
                results = [(max([np.add.reduce(roi_mask[max([int(y2)-1, 0]), int(x1):int(x2)].reshape((-1,))) for x1, _, x2, y2 in result.tolist()]) if result.shape[0]>0 else 0) for result in results] 
                
                # find max roi intersection for this image
                results = max(results) if len(results)>0 else 0
                if results>0:
                    logger.info("Humans Entered into Region of Interest.")

                # list of tuples of (optional(ndarray), int)
                dec_window_list_imgresults.append((img if results > 0 else None, results)) 
                





                # remove older data if excess
                while(len(dec_window_list_imgresults)>dec_window_size):
                    dec_window_list_imgresults.pop(0)
                    
                
                if not has_seen: # only trigger sending mechanism if old detection is not ongoing
                    # greater than 0 if overlap exits
                    if (sum([1 if i>0 else 0 for _, i in dec_window_list_imgresults]) >= dec_window_approv):
                        # Find image with the largest overlap with ROI
                        imgr, _ = max(dec_window_list_imgresults, key=lambda x: x[1])
                        # Send image for printing
                        if imgr is not None: # safety --> will only be an issue if `dec_window_approv==0`
                            logger.info("Human Alert sent; Alert deactivated.")
                            printing_images_q.put((camera_TID, imgr)) 
                            has_seen = True
                else: # if an old detection has been sent
                    if (sum([i for _, i in dec_window_list_imgresults])<=dec_window_release): # no detections triggered in last 3 frames
                        # dec_window_list_imgresults = dec_window_list_imgresults[-3:] # start afresh; keeping last 3 frames
                        logger.info("Human has left Region of Interest; Alert activated.")
                        has_seen = False

                        
                        
                del img
                del camera_TID
        except Exception as e:
            print(e)
            elegant_shutdown.put(True)
    elegant_shutdown.put(True)


def ImageCapture_IO():
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='./LOGS/FILES/ImageCapture.logs', encoding='utf-8', level=logging.INFO)

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
                        logger.warning("Grab Failure")
                        cap.release()
                        logger.warning("Released VideoCapture")
                        cap = cv.VideoCapture(cameras_link)
                        logger.warning("Re-init VideoCapture")
                        continue
                    else:
                        logger.error("Grab Failure")
                        raise ValueError("Grab Failure")

                if (count%(frame_const//5) == 0): # keeping it to 5 frames per second or less
                    ret, frame = cap.retrieve()
    
                    if (not ret):
                        logger.error("Retrieve Failure")
                        raise ValueError("Retrieve Failure")


                    if ret and (capture_images_q.empty() and capture_images_f.empty()):
                        recover = 0
                        dtm_ = datetime.datetime.now(pytz.utc).isoformat().split('+')[0]
                        # print(f"Sent Successful:\t{count}")
                        logger.info(f"Read Frame: {count}:{dtm_}; Sent to Model")
                        capture_images_q.put((f"{dtm_}@{cameras_id}", frame[:,:,:]))
                        capture_images_f.put((f"{dtm_}@{cameras_id}", frame[:,:,:]))
                        del frame



                time.sleep(1/fpso)
        except Exception as e:
            print(e)
        finally:
            cap.release()
            elegant_shutdown.put(True)




def main():

    if not(os.path.exists("./LOGS/") and os.path.isdir("./LOGS/")):
        os.mkdir("./LOGS/")
    if not(os.path.exists("./LOGS/HUMAN/") and os.path.isdir("./LOGS/HUMAN/")):
        os.mkdir("./LOGS/HUMAN/")
    if not(os.path.exists("./LOGS/FIRE/") and os.path.isdir("./LOGS/FIRE/")):
        os.mkdir("./LOGS/FIRE/")
    if not(os.path.exists("./LOGS/FILES/") and os.path.isdir("./LOGS/FILES/")):
        os.mkdir("./LOGS/FILES/")


    parser = argparse.ArgumentParser(description='Watch Cameras for Humans')
    parser.add_argument('--rtsp', type=str, help='rtsp link for camera', default=None)
    parser.add_argument('--fpath', type=str, help='rtsp link for camera', default=None)
    parser.add_argument('--uuid', type=str, help='rtsp link for camera', default=None)
    parser.add_argument('--env_camera', type=int, help='index of camera from .env.json file', default=0)
    args = parser.parse_args()

    cameras = None
    if (args.env_camera is not None):
        with open("./.env.json", 'r') as env_file:
            jdata = json.load(env_file)
            cameras = jdata["cameras"] if "cameras" in jdata else []
        cameras_links.put(cameras[args.env_camera])

    else:
        if ((args.uuid is not None) and (not (args.rtsp is None or args.fpath is None))):
            cameras_links.put({
                    "uid": args.uuid,
                    "link": (args.rtsp if args.rtsp is not None else args.fpath)
                })
        else:
            print("(--rtsp or --fpath) and --uuid cannot be empty")
            exit()
    
    

    
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



