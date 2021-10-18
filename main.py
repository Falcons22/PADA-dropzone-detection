import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import webcolors

#finding circles in the frame and drawing circles
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

#approximation to closest colour name
def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS21_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

#finding the color
def findColor(circles1,img_rgb):
    l=[]
    if circles1 is not None:
        for i in circles1:
            color=tuple(img_rgb[i[1],i[0],:])
            color=(color[2],color[0],color[1])
            try:
                name=webcolors.rgb_to_name(color)
                l.append(name)
            except:
                l.append(closest_colour(color))
    return " ".join(elem for elem in l)



#initiating camera
cap = cv2.VideoCapture(0) 
# last frame time
prev_frame_time = 0
 
# current frame time
new_frame_time = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    #calculating fps
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #changing image to gray scale
    img_blur= cv2.GaussianBlur(gray, (3,3), 0) #gaussian blur
    #img_blur=cv2.blur(gray,(3,3)) (for box blur)
    img1,circles1=findCircles(frame,img_blur)

    #putting text of color and fps on image
    cv2.putText(img1, str(int(fps)), (7,470), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(img1, findColor(circles1,frame), (500,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow('Test',img1) #showing image on frame
    #press q to end
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
