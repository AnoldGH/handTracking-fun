import cv2
import mediapipe as mp
import time

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
    
    # Drawing Sub-module
    def _track_landmark_safe(self, img, hdID, lmID, radius, color, thickness=1, linetype=cv2.LINE_8):
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hdID]
            lm = hand.landmark[lmID]
            cx, cy = int(self.width * lm.x), int(self.height * lm.y)
            cv2.circle(img, (cx, cy), radius, color, thickness, linetype)
    
    def _connect_landmarks_safe(self, img, hdID, lm1ID, lm2ID, color, thickness=1, linetype=cv2.LINE_8):
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hdID]
            lm1, lm2 = hand[lm1ID], hand[lm2ID]
            cv2.line(img, (lm1.x, lm1.y), (lm2.x, lm2.y), \
                color, thickness, linetype)
    
    def track_landmark(self, img, hdID, lmID, radius, color, thickness=1, linetype=cv2.LINE_8):
        curry = lambda: self._track_landmark_safe(img, hdID, lmID, radius, color, thickness, linetype)
        self.drawings.append(curry)    
        
    def connect_landmarks(self, img, hdID, lm1ID, lm2ID, color, thickness=1, linetype=cv2.LINE_8):
        curry = lambda: \
            self._connect_landmarks_safe(img, hdID, lm1ID, lm2ID, color, thickness, linetype)
        self.drawings.append(curry)
    
    def render(self):
        for drawing in self.drawings:
            drawing()
    
    
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
            detector.track_landmark(img, 0, 4, 15, (0, 255, 255))
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