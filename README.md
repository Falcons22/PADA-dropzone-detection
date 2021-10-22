# PADA-dropzone-detection
### To run this install the dependencies using
```
pip install -r requirements.txt
```

### To run this use the following command lines
```
python main.py
```

In this code I have captured the image, then changed it into gray scan and blurred the image using gaussian blur. Then used Hough circles function to identify the circles and have drawn circles over it. Then found the color of the center of the circle.

Another method is with color masking. To run this use
```
python detection_with_masking.py
```
In this method, the limits must be adjusted to find the required color and then the circle.
### Preview

<p align="center">
  <img src="demo.png">
  <p align="center">
  Circle detection</p>
 </p>
<p align="center">
  <img src="demo_masking.png">
  Circle detection with color masking (blue)
 </p>
 <p align="center">
  <img src="demo_masking_red.png">
  Circle detection with color masking (red)
 </p>
