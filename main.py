import cv2
import mediapipe as mp
import pickle
import numpy as np

from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display

model = pickle.load(open("model.pkl", "rb"))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("Sign Language Translator", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Sign Language Translator", 1280, 720)

predictions = []

translation = {
    "HELLO": "مرحبا",
    "YES": "نعم",
    "NO": "لا",
    "THUMBSUP": "ممتاز",
    "ONE": "واحد",
    "TWO": "اثنين"
}

font = ImageFont.truetype("arial.ttf", 32)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x_list = []
            y_list = []
            data = []

            for lm in hand_landmarks.landmark:
                x_list.append(lm.x)
                y_list.append(lm.y)
                data.append(lm.x)
                data.append(lm.y)

            xmin = int(min(x_list) * w)
            xmax = int(max(x_list) * w)
            ymin = int(min(y_list) * h)
            ymax = int(max(y_list) * h)

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0), 2)

            proba = model.predict_proba([np.array(data)])
            prediction = model.predict([np.array(data)])

            confidence = np.max(proba)
            confidence_percent = int(confidence * 100)

            if confidence > 0.5:
                predictions.append(prediction[0])

            if len(predictions) > 15:
                predictions.pop(0)

            counts = {}
            for i, pred in enumerate(predictions):
                weight = i + 1
                counts[pred] = counts.get(pred, 0) + weight

            final_pred = max(counts, key=counts.get) if counts else ""

            if confidence_percent < 60:
                final_pred = ""
                text = ""
            else:
                arabic_word = translation.get(final_pred, "")

                if arabic_word != "":
                    reshaped = arabic_reshaper.reshape(arabic_word)
                    bidi_text = get_display(reshaped)
                    text = f"{final_pred} ({bidi_text}) {confidence_percent}%"
                else:
                    text = f"{final_pred} {confidence_percent}%"

            img_pil = Image.fromarray(frame)
            draw = ImageDraw.Draw(img_pil)

            text_x = xmin
            text_y = ymin - 40 if ymin - 40 > 0 else ymin + 10

            draw.text((text_x, text_y), text, font=font, fill=(0,255,0))

            frame = np.array(img_pil)

    else:
        cv2.putText(frame, "No hand detected", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Sign Language Translator", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()