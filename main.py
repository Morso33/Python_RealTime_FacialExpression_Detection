import cv2, os
import numpy as np
from PIL import Image

#Setup webcam
print("starting webcam, please wait...")
cap = cv2.VideoCapture(0)
print("ready!")
if not cap.isOpened():
    print("Error opening video stream")
# Loop until 'esc' is pressed
while True:
    ret, frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)
    cv2.putText(frame, "FPS: " + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    #Get the webcam frame and convert it to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Frame', gray)

    if cv2.waitKey(1) == 27: #ESC
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
