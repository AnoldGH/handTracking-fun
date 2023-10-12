import cv2
import mediapipe as mp
import time
import math
import numpy as np

# Macros
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_FINGER_MCP = 5
INDEX_FINGER_PIP = 6
INDEX_FINGER_DIP = 7
INDEX_FINGER_TIP = 8
MIDDLE_FINGER_MCP = 9
MIDDLE_FINGER_PIP = 10
MIDDLE_FINGER_DIP = 11
MIDDLE_FINGER_TIP = 12
RING_FINGER_MCP = 13
RING_FINGER_PIP = 14
RING_FINGER_DIP = 15
RING_FINGER_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

class handDetector():
      
    
    def __init__(self, running_mode=False, num_hands=2, model_complexity=1, min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5, result_callback=None) -> None:
        
        self.running_mode = running_mode
        self.num_hands = num_hands
        self.model_complexity = model_complexity
        self.min_hand_detection_confidence = min_hand_detection_confidence
        self.min_hand_presence_confidence = min_hand_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.result_callback = result_callback
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(running_mode, num_hands, model_complexity, min_hand_detection_confidence, min_tracking_confidence)
        
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None
        
        # Drawing Module
        self.drawings = list()
        
        # todo: parametrize the following
        self.height = 360
        self.width = 640
        
        # Stats - used for running diagnostics
        self._stime = time.time()
        self._ptime = self._stime
        self._ctime = self._stime
        self.fps = None
        
    def findHands(self, img, draw=True):
        # self.results = self.hands.process(imgRGB)
        
        if self.results.multi_hand_landmarks:
            for handLandmark in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLandmark, self.mpHands.HAND_CONNECTIONS)
                    
        return img
    
    def findPosition(self, img, handId=0, draw=True):
        landmarks = list()
        
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[handId]
            
            for id, lm in enumerate(hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int (lm.y * h)
                landmarks.append([id, cx, cy])
                
                if draw: 
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        
        return landmarks
    
    # Returns False if failed to detect any hand, otherwise True
    def update(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        
        self._ctime = time.time()
        self.fps = int(1 / (self._ctime - self._ptime))
        self._ptime = self._ctime
        
        return self.results.multi_hand_landmarks is not None
        
    
    def getFPS(self):
        return self.fps      
    
    
    # Utility Functions
    def distance_between(self, hd1ID, lm1ID, hd2ID, lm2ID):
        cx1, cy1 = self.positionOf(lm1ID, hd1ID)
        cx2, cy2 = self.positionOf(lm2ID, hd2ID)
        if cx1 is not None and cx2 is not None:
            return math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
        return None
    
    
    # Drawing Sub-module
    def _track_landmark_safe(self, img, hdID, lmID, radius, color, thickness=2, linetype=cv2.LINE_8):
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hdID]
            lm = hand.landmark[lmID]
            cx, cy = int(self.width * lm.x), int(self.height * lm.y)
            cv2.circle(img, (cx, cy), radius, color, thickness, linetype)            
    
    def _track_landmarks_connection_safe(self, img, hdID, lm1ID, lm2ID, color, thickness=2, linetype=cv2.LINE_8):
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hdID]
            lm1, lm2 = hand.landmark[lm1ID], hand.landmark[lm2ID]
            cx1, cy1 = int(self.width * lm1.x), int(self.height * lm1.y)
            cx2, cy2 = int(self.width * lm2.x), int(self.height * lm2.y)
            cv2.line(img, (cx1, cy1), (cx2, cy2), \
                color, thickness, linetype)

    # Helper functions
    def _midpoint(self, *pos):
        pos1 = pos[0]
        pos2 = pos[1]
        return (pos1 + pos2) / 2   
    
    @DeprecationWarning
    def __calculate_safe(self, func, pos1, pos2):
        if pos1 is None or pos2 is None:
            return None
        else: return func(pos1, pos2)
        
    def _calculate_safe(self, func, *param_list):
        for param in param_list:
            if param is None: return None
        return func(*param_list).astype(int)
    
    def track_landmark(self, hdID, lmID, radius, color, thickness=1, linetype=cv2.LINE_8):
        curry = lambda img: self._track_landmark_safe(img, hdID, lmID, radius, color, thickness, linetype)
        self.drawings.append(curry)
        # Return a curry function to get landmark coordinate
        return lambda: self.positionOf(lmID, hdID)
        
    def track_landmarks_connection(self, hdID, lm1ID, lm2ID, color, thickness=1, linetype=cv2.LINE_8):
        curry = lambda img: \
            self._track_landmarks_connection_safe(img, hdID, lm1ID, lm2ID, color, thickness, linetype)
        self.drawings.append(curry)
        # Return a curry function to get two end-points
        return lambda: (self.positionOf(lm1ID, hdID), self.positionOf(lm1ID, hdID))
        
    def track_midpoint_between(self, hd1ID, lm1ID, hd2ID, lm2ID, radius, color, thickness=1, linetype=cv2.LINE_8):
        query = lambda: self._calculate_safe(
            self._midpoint, self.positionOf(lm1ID, hd1ID), 
            self.positionOf(lm2ID, hd2ID))
        
        def curry(img):
            tpos = query()
            if tpos is not None:
                cv2.circle(img, tpos, radius, color, thickness, linetype)
        self._draw_custom_point(curry)

        # Return a curry function for value querying
        return query
    
    def _draw_custom_point(self, drawing):
        self.drawings.append(drawing)
        
    def render(self, img):
        for drawing in self.drawings:
            drawing(img)
    
    # Curry helper functions
    def Xof(self, lmID, hdID=0):
        hand = self.results.multi_hand_landmarks
        if hand: return self.width * hand[hdID].landmark[lmID].x
        else: return None
    
    def Yof(self, lmID, hdID=0):
        hand = self.results.multi_hand_landmarks
        if hand: return self.height * hand[hdID].landmark[lmID].y
        else: return None
    
    def positionOf(self, lmID, hdID=0, cast_to_int=True):
        x = self.Xof(lmID, hdID)
        y = self.Yof(lmID, hdID)
        if x is None or y is None: return None
        else: 
            pos = np.array([self.Xof(lmID, hdID), self.Yof(lmID, hdID)])
            if cast_to_int: pos = pos.astype(int)
            return pos

    
def main():
    capture = cv2.VideoCapture(0)
    detector = handDetector()
    
    video = cv2.VideoWriter('video.mp4', -1, 24, (640, 360))
    
    # Drawing
    detector.track_landmark(0, PINKY_TIP, 15, (0, 255, 255))
    detector.track_landmarks_connection(0, MIDDLE_FINGER_TIP, WRIST, (255, 0, 0))
    detector.track_midpoint_between(0, PINKY_TIP, 0, MIDDLE_FINGER_TIP, 5, (255, 0, 0), 3, cv2.FILLED)
    
    i = 0
    while True:
        success, img = capture.read()
        if success:
            detector.update(img)
            
            img = detector.findHands(img)
            
            cv2.putText(img, str(detector.getFPS()), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            
            # video.write(img)
            
            
            detector.render(img)
            
            cv2.imshow("Image", img)
            
            cv2.waitKey(1)
        else:
            print('Unable to read from camera')
            break
        
        i += 1
        
    video.release()
            
if __name__ == '__main__':
    main()