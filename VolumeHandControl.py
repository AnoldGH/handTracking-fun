import cv2
import time
import numpy as np
import HandTrackingModule as handTrack
import math

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


# Macros
webcam_id = 0
wCam, hCam = 640, 480
fpsScale, fpsThickness = 1, 2
fpsColor = [255, 0, 0]

capture = cv2.VideoCapture(webcam_id)
capture.set(3, wCam)
capture.set(4, hCam)
pTime = 0

detector = handTrack.handDetector(min_hand_detection_confidence=0.7)

# Tracker
thumb_tip = detector.track_landmark(
    0, handTrack.THUMB_TIP, 15, (255, 0, 255), 1, cv2.FILLED)
index_tip = detector.track_landmark(
    0, handTrack.INDEX_FINGER_TIP, 15, (255, 0, 255), 1, cv2.FILLED)
detector.track_landmarks_connection(
    0, handTrack.THUMB_TIP, handTrack.INDEX_FINGER_TIP, 
    (255, 0, 255), 1, cv2.FILLED)
thumb_index_midpoint = detector.track_midpoint_between(0, handTrack.THUMB_TIP, 0, handTrack.INDEX_FINGER_TIP, 15, (0, 255, 255), 2, cv2.FILLED)

# Volume Control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
minVolume, maxVolume, _ = volume.GetVolumeRange()
# volume.SetMasterVolumeLevel(-20.0, None)
vol = 0
volBar = 400


while True:
    success, img = capture.read()
    
    if success:
        hasHand = detector.update()
        img = detector.findHands(img)

        if hasHand:
            index_tip_pos = index_tip()
            thumb_tip_pos = thumb_tip()
            length = math.hypot(*(index_tip_pos - thumb_tip_pos))
            
            if length < 50:
                cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
                
            vol = np.interp(length, [50, 300], [minVolume, maxVolume])
            volBar = np.interp(length, [50, 300], [400, 150])
            volume.SetMasterVolumeLevel(vol, None)
            
        detector.render()
    
    # landmarks = detector.findPosition(img, draw=False)
    # if len(landmarks) != 0:
    #     x1, y1 = landmarks[4][1], landmarks[4][2]
    #     x2, y2 = landmarks[8][1], landmarks[8][2]
    #     cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
    #     cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
    #     cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
    #     cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
    #     cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        
    #     length = math.hypot(x2 - x1, y2 - y1)
        
    #     if length < 50:
    #         cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

    #     vol = np.interp(length, [50, 300], [minVolume, maxVolume])
    #     volBar = np.interp(length, [50, 300], [400, 150])
    #     volume.SetMasterVolumeLevel(vol, None)

    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
    
    # Display fps
    cv2.putText(img, f'FPS: {int(detector.getFPS())}', (40, 70), cv2.FONT_HERSHEY_COMPLEX, fpsScale, fpsColor, fpsThickness)
    
    cv2.imshow("Img", img)
    cv2.waitKey(1)
