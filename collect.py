import cv2
import mediapipe as mp
import pickle
import os
import time  

label = input("Enter label: ")  # HELLO / YES

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

data = []
count = 0


while count < 500:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            row = []
            for lm in hand_landmarks.landmark:
                row.append(lm.x)
                row.append(lm.y)

            data.append(row)
            count += 1

            time.sleep(0.1)  

    
    cv2.putText(frame, f"{label}: {count}/300", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Collecting Data", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

#دمج البيانات
filename = f"{label}.pkl"

if os.path.exists(filename):
    old_data = pickle.load(open(filename, "rb"))
    data = old_data + data

with open(filename, "wb") as f:
    pickle.dump(data, f)

print(f"✅ Saved {label} with {len(data)} samples!")