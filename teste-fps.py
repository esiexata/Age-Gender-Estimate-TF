import cv2
import numpy as np

cap = cv2.VideoCapture(0)
fps=5 #for decreasing fps

while True:
    ret,frame=cap.read()
    if fps<2:
        cv2.imshow('frame drop',frame)
        fps=0
        fps+=1
        if(cv2.waitKey(1) & 0xFF==ord('q')):
            break

cap.release()
cv2.destroyAllWindows()