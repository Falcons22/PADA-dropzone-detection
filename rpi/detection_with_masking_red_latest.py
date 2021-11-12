# %%
import cv2
import numpy as np

# %%
def findCircles(img,img_blur):
    output=img.copy()
    #cv2.blur(gray,(3,3))
    # high_thresh, thresh_im = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # lowThresh = 0.5*high_thresh
    circles = cv2.HoughCircles(img_blur,cv2.HOUGH_GRADIENT,3,1000,minRadius=0)
            # param1=high_thresh,param2=lowThresh,
            # minRadius=50, maxRadius=400)
                #,minRadius=20)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
    return output,circles


# %%
cap = cv2.VideoCapture("video.mp4")
while True:
    ret, frame = cap.read()

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Red color

    low_red = np.array([10, 127, 56])
    high_red = np.array([179, 255, 255])
    red_mask = cv2.inRange(hsv_frame, low_red, high_red)
    edges = cv2.Canny(image=red_mask, threshold1=100, threshold2=50)
    cv2.namedWindow('Red', cv2.WINDOW_NORMAL)
    cv2.imshow("Red", red_mask)
    try:
        result=findCircles(frame,edges)
        cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
        cv2.imshow("Result", result[0])
    except:
        pass
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
