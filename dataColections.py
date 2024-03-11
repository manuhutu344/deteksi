import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math

cap = cv2.VideoCapture(0)
ditector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

while True:
    success, img = cap.read()
    hands, img = ditector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        imgWhite = np.ones((imgSize, imgSize,3), np.uint8)*255
        imgCrop = img[y-offset:y + h+offset, x-offset:x + w+offset]
        imgCropShape = imgCrop.shape
        

        aspectRation = h/w
        if aspectRation >1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgReize = cv2.resize(imgCrop, (wCal, imgSize))
            imgReiszeShape = imgReize.shape
            imgWhite[0:imgReiszeShape[0],0:imgReiszeShape[1]] = imgReize


        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    cv2.waitKey(1)



