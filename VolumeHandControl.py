import cv2
import time
import numpy as np
import HandTrackingModule as handTrack

# Macros
webcam_id = 0
wCam, hCam = 640, 480
fpsScale, fpsThickness = 1, 2
fpsColor = list(255, 0, 0)

capture = cv2.VideoCapture(webcam_id)
capture.set(3, wCam)
capture.set(4, hCam)
pTime = 0

detector = handTrack.handDetector(min_hand_detection_confidence=0.7)


while True:
    success, img = capture.read()
    img = detector.findHands(img)
    landmarks = detector.findPosition(img, draw=False)
    
    # Calculate fps
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    # Display fps
    cv2.putText(img, f'FPS: {int(fps)}', (40, 70), cv2.FONT_HERSHEY_COMPLEX, fpsScale, fpsColor, fpsThickness)
    
    cv2.imshow("Img", img)
    cv2.waitKey(1)
    
