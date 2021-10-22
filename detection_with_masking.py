import cv2
import numpy as np
import time


def findCircles(img,img_blur):
    output=img.copy()
    #cv2.blur(gray,(3,3))
    # high_thresh, thresh_im = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # lowThresh = 0.5*high_thresh
    circles = cv2.HoughCircles(img_blur,cv2.HOUGH_GRADIENT, 1.2,1000,)
            # param1=high_thresh,param2=lowThresh,
            # minRadius=20, maxRadius=400)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)#drawing circle
    return output,circles

def nothing(x):
    pass

cv2.namedWindow('marking')

cv2.createTrackbar('H Lower','marking',0,179,nothing)
cv2.createTrackbar('H Higher','marking',179,179,nothing)
cv2.createTrackbar('S Lower','marking',0,255,nothing)
cv2.createTrackbar('S Higher','marking',255,255,nothing)
cv2.createTrackbar('V Lower','marking',0,255,nothing)
cv2.createTrackbar('V Higher','marking',255,255,nothing)


cap = cv2.VideoCapture(0)
prev_frame_time = 0
 
# current frame time
new_frame_time = 0
while(1):
    _, frame = cap.read()
    # It converts the BGR color space of image to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    #     blue = np.uint8([[[255,0, 0]]]) 
    #     hsvBlue = cv2.cvtColor(blue, cv2.COLOR_BGR2HSV)
    #print(hsvBlue)
    # lowerLimit = hsvBlue[0][0][0] - 10, 100, 100
    # lowerLimit = np.array(lowerLimit)
    # upperLimit = hsvBlue[0][0][0] + 10, 255, 255
    # upperLimit = np.array(upperLimit)
    # print(lowerLimit)
    hL = cv2.getTrackbarPos('H Lower','marking')
    hH = cv2.getTrackbarPos('H Higher','marking')
    sL = cv2.getTrackbarPos('S Lower','marking')
    sH = cv2.getTrackbarPos('S Higher','marking')
    vL = cv2.getTrackbarPos('V Lower','marking')
    vH = cv2.getTrackbarPos('V Higher','marking')

    LowerRegion = np.array([hL,sL,vL],np.uint8)
    upperRegion = np.array([hH,sH,vH],np.uint8)

    # lower_blue = np.array([40, 35, 140])
    # upper_blue = np.array([180, 255, 255])
    # # preparing the mask to overlay
    mask = cv2.inRange(hsv, LowerRegion, upperRegion)
    
    result = cv2.bitwise_and(frame,frame, mask = mask)
    img_blur=cv2.GaussianBlur(result,(3,3),0)
    result=cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.putText(result, str(int(fps)), (7,470), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)
    result=findCircles(frame,result)
    cv2.imshow('result', result[0])
     
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cv2.destroyAllWindows()
cap.release()
