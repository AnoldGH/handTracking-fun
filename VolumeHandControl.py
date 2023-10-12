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
    0, handTrack.THUMB_TIP, handTrack.INDEX_FINGER_TIP, (255, 0, 255))
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

video = cv2.VideoWriter('video.mp4', -1, 24, (640, 360))

i = 0
while i < 150:
    success, img = capture.read()
    
    if success:
        hasHand = detector.update(img)

        if hasHand:
            index_tip_pos = index_tip()
            thumb_tip_pos = thumb_tip()
            thumb_index_mp_pos = thumb_index_midpoint()
            length = math.hypot(*(index_tip_pos - thumb_tip_pos))
            
            # Debug Info
            print(index_tip_pos)
            print(thumb_tip_pos)
            print(thumb_index_mp_pos)
            print(length)
            
            if length < 50:
                cv2.circle(img, thumb_index_mp_pos, 15, (0, 255, 0), cv2.FILLED)
                
            vol = np.interp(length, [50, 300], [minVolume, maxVolume])
            volBar = np.interp(length, [50, 300], [400, 150])
            volume.SetMasterVolumeLevel(vol, None)
            
        detector.render(img)
        video.write(img)

    # Display Volume Bar
    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
    
    # Display FPS
    cv2.putText(img, f'FPS: {int(detector.getFPS())}', (40, 70), cv2.FONT_HERSHEY_COMPLEX, fpsScale, fpsColor, fpsThickness)
    
    cv2.imshow("Img", img)
    cv2.waitKey(1)
    
    i += 1

video.release()