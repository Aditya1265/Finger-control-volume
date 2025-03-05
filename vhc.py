import cv2
import mediapipe as mp
import numpy as np
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)


vol_range = volume.GetVolumeRange()
min_vol, max_vol = vol_range[0], vol_range[1]


cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

           
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = img.shape
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            if len(lm_list) >= 8:
                x1, y1 = lm_list[4]  
                x2, y2 = lm_list[8]  

              
                cv2.circle(img, (x1, y1), 10, (255, 0, 0), -1)
                cv2.circle(img, (x2, y2), 10, (255, 0, 0), -1)
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

                length = np.linalg.norm(np.array([x2, y2]) - np.array([x1, y1]))

                
                vol = np.interp(length, [30, 200], [min_vol, max_vol])  
                volume.SetMasterVolumeLevel(vol, None)

                
                vol_percent = np.interp(length, [30, 200], [0, 100])
                cv2.putText(img, f'Volume: {int(vol_percent)}%', (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("Hand Gesture Volume Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break

cap.release()
cv2.destroyAllWindows()
