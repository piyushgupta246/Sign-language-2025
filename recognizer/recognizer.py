import cv2
import mediapipe as mp
import pickle
import numpy as np
from collections import deque

class SimpleSignLanguageRecognizer:
    def __init__(self, model_path="model/simple_sign_model.pkl"):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)
        self.prediction_history = deque(maxlen=5)

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                landmarks = []
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)

                for landmark in hand_landmarks.landmark:
                    landmarks.extend([
                        (landmark.x - min_x) / (max_x - min_x + 1e-6),
                        (landmark.y - min_y) / (max_y - min_y + 1e-6)
                    ])

                prediction = self.model.predict([landmarks])[0]
                self.prediction_history.append(prediction)

                if len(self.prediction_history) >= 3:
                    prediction = max(set(self.prediction_history),
                                     key=self.prediction_history.count)

                x1 = int(min_x * frame.shape[1])
                y1 = int(min_y * frame.shape[0])
                x2 = int(max_x * frame.shape[1])
                y2 = int(max_y * frame.shape[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, prediction, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        return frame
