import cv2
import mediapipe as mp
import time


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
        self.stime = time.time()
        self.ptime = self.stime
        self.ctime = self.stime
        self.fps = None
        
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        
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
    
    def update(self, img):
        self.results = self.hands.process(img, cv2.COLOR_BGR2RGB)
        self.ctime = time.time()
        self.fps = int(1 / (self.ctime - self.ptime))
        self.ptime = self.ctime
    
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
    
    def track_landmark(self, img, hdID, lmID, radius, color, thickness=1, linetype=cv2.LINE_8):
        curry = lambda: self._track_landmark_safe(img, hdID, lmID, radius, color, thickness, linetype)
        self.drawings.append(curry)
        
    def track_landmarks_connection(self, img, hdID, lm1ID, lm2ID, color, thickness=1, linetype=cv2.LINE_8):
        curry = lambda: \
            self._track_landmarks_connection_safe(img, hdID, lm1ID, lm2ID, color, thickness, linetype)
        self.drawings.append(curry)
        
    def track_custom_point(self, drawing):
        self.drawings.append(drawing)
        
    def render(self):
        for drawing in self.drawings:
            drawing()
    
    # Curry helper functions
    def Xof(self, lmID):
        def getter():
            hand = self.results.multi_hand_landmarks
            if hand: return self.width * hand.landmark[lmID].x
            else: return None
        return getter
    
    def Yof(self, lmID):
        def getter():
            hand = self.results.multi_hand_landmarks
            if hand: return self.height * hand.landmark[lmID].y
            else: return None
        return getter
    
    def positionOf(self, lmID):
        return self.Xof(lmID), self.Yof(lmID)
    

    
def main():
    pTime = 0
    cTime = 0
    capture = cv2.VideoCapture(0)
    detector = handDetector()
    
    video = cv2.VideoWriter('video.mp4', -1, 24, (640, 360))
    
    i = 0
    while True:
        success, img = capture.read()
        if success:
            img = detector.findHands(img)
            
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            
            cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            
            # video.write(img)
            
            # Drawing
            detector.track_landmark(img, 0, PINKY_TIP, 15, (0, 255, 255))
            detector.track_landmarks_connection(img, 0, MIDDLE_FINGER_TIP, WRIST, (255, 0, 0))
            
            # Example custom function
            itX, itY = detector.positionOf(INDEX_FINGER_TIP)
            def track_middle_point_between_indextip_pinkytip():
                cv2.circle(img, (int(itX()), int(itY())), 5, (255, 0, 0), 3)
            detector.track_custom_point(track_middle_point_between_indextip_pinkytip)
            
            detector.render()
            
            cv2.imshow("Image", img)
            
            cv2.waitKey(1)
        else:
            print('Unable to read from camera')
            break
        
        i += 1
        
    video.release()
            
if __name__ == '__main__':
    main()