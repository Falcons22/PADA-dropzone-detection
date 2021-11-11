import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from webcolors import rgb_to_name


def findCircles(img,img_blur):
    output=img.copy()
    #cv2.blur(gray,(3,3))
    # high_thresh, thresh_im = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # lowThresh = 0.5*high_thresh
    circles = cv2.HoughCircles(img_blur,cv2.HOUGH_GRADIENT, 1.2,1000,)
            # param1=high_thresh,param2=lowThresh,
            # minRadius=50, maxRadius=400)
                #,minRadius=20)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
    return output,circles

def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def findColor(circles1,img_rgb):
    l=[]
    if circles1 is not None:
        for i in circles1:
            try:
                name=rgb_to_name(tuple(img_rgb[i[1],i[0],:]))
                l.append(name)
            except:
                l.append(RGB2HEX(tuple(img_rgb[i[1],i[0],:])))
                #print(l)
    return " ".join(elem for elem in l)


cap = cv2.VideoCapture(0) 
# used to record the time when we processed last frame
prev_frame_time = 0
 
# used to record the time at which we processed current frame
new_frame_time = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_blur_gaussian= cv2.GaussianBlur(gray, (3,3), 0)
    #img_blur_gaussian1=cv2.Canny(gray, 100, 50, 3 )
    img1,circles1=findCircles(frame,img_blur_gaussian)

    cv2.putText(img1, str(int(fps)), (7,470), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(img1, findColor(circles1,frame), (500,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow('Test',img1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()