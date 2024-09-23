import numpy as np
import cv2 as cv


def create_labels(img, mask_color=(0,0,0)):

    if (img is not None):        
        points = []
        bins = []
            
        def draw_circle(event,x,y,flags,param):
            if event == cv.EVENT_LBUTTONDBLCLK:
                
                if len(points)<4:
                    cv.circle(img,(x,y),5,(255,0,0),-1)
                    points.append((x,y))
                else:
                    cv.circle(img,(x,y),5,(0,255,0),-1)
                    bins.append((x,y))


        cv.namedWindow("Image", cv.WINDOW_NORMAL)
        cv.setMouseCallback("Image", draw_circle)
        while True:
            img_c = img.copy()
            img_c = cv.fillPoly(img.copy(), [cv.convexHull(np.array(points), returnPoints=True).reshape((-1,2))], mask_color) if len(points)==4 else img_c
            for bin in bins:
                cv.circle(img_c, bin,5,(0,255,0),-1)

            cv.imshow("Image", img_c)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
        cv.destroyAllWindows()

        # return (cv.convexHull(np.array(points), returnPoints=True).reshape((-1,2)), bins)
        return (np.array(points), bins)

##! Can help create the layout for the dustbin placement