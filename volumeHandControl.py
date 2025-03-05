import cv2 
import time
import numpy as np
import HandTrackingModule as htm
import os
import tensorflow as tf
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
wcam,hcam= 640,480

cap=cv2.VideoCapture(0)
cap.set(3, wcam)
cap.set(4,hcam)
Ptime=0

detector= htm.handDetector()

while True:
    success ,img =cap.read()
    img= detector.findHands(img)


    ctime=time.time()
    fps=1/(ctime-Ptime)
    Ptime= ctime
    cv2.putText(img,f'FPS: {int(fps)}',(40,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)
    cv2.imshow("img",img)
    cv2.waitKey(1)     #1ms delay