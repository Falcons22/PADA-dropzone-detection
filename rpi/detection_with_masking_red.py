# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import cv2
import numpy as np
import time

# %%
def findCircles(img,img_blur):
    output=img.copy()
    #cv2.blur(gray,(3,3))
    # high_thresh, thresh_im = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # lowThresh = 0.5*high_thresh
    circles = cv2.HoughCircles(img_blur,cv2.HOUGH_GRADIENT, 3,1000,)
            # param1=high_thresh,param2=lowThresh,
            # minRadius=50, maxRadius=400)
                #,minRadius=20)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
    return output,circles


# %%
#def nothing(x):
    #pass

#cv2.namedWindow('marking')

#cv2.createTrackbar('H Lower','marking',0,179,nothing)
#cv2.createTrackbar('H Higher','marking',179,179,nothing)
#cv2.createTrackbar('S Lower','marking',0,255,nothing)
#cv2.createTrackbar('S Higher','marking',255,255,nothing)
#cv2.createTrackbar('V Lower','marking',0,255,nothing)
#cv2.createTrackbar('V Higher','marking',255,255,nothing)


# %%
# used to record the time when we processed last frame
prev_frame_time = 0
 
# used to record the time at which we processed current frame
new_frame_time = 0
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Red color
    #hL = cv2.getTrackbarPos('H Lower','marking')
    #hH = cv2.getTrackbarPos('H Higher','marking')
    #sL = cv2.getTrackbarPos('S Lower','marking')
    #sH = cv2.getTrackbarPos('S Higher','marking')
    #vL = cv2.getTrackbarPos('V Lower','marking')
    #vH = cv2.getTrackbarPos('V Higher','marking')
    #low_red = np.array([10, 127, 255])
    #high_red = np.array([179, 255, 255])
    low_red=np.array([3,85,75])
    high_red=np.array([179,255,255])
    red_mask = cv2.inRange(hsv_frame, low_red, high_red)
    #red = cv2.bitwise_and(frame, frame, mask=red_mask)
    #red_blur=cv2.GaussianBlur(red,(5,5),0)
    #result,circles=findCircles(red,red_blur)
    cv2.imshow("Red", red_mask)
    #cv2.imshow("Red Blur", red_blur)
    #red_gray=cv2.cvtColor(red_blur,cv2.COLOR_HSV2BGR)
    #red_gray=cv2.cvtColor(red_gray,cv2.COLOR_BGR2GRAY)
    try:
        result=findCircles(frame,red_mask)
        
        cv2.imshow("Result", result[0])
    except:
        pass
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()


# %%



