import os
import pickle
import numpy as np
import cv2
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define our target words/phrases
TARGET_WORDS = {
    0: "Hey",
    1: "Hello",
    2: "Good Morning",
    3: "Bye",
    4: "See You",
    5: "Thank You",
    6: "Please"
}

class SimpleSignLanguageTrainer:
    def __init__(self):
        self.data_dir = "simple_sign_data"
        self.model_path = "model/simple_sign_model.pkl"   # <== model saved inside 'model/' folder
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.data = []
        self.labels = []

        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

    def collect_samples(self, samples_per_class=50):
        print(f"Collecting {samples_per_class} samples for each word/phrase")
        print("Available words:", TARGET_WORDS.values())

        cap = cv2.VideoCapture(0)

        for word_id, word in TARGET_WORDS.items():
            word_dir = os.path.join(self.data_dir, word)
            os.makedirs(word_dir, exist_ok=True)

            existing_samples = len([f for f in os.listdir(word_dir) if f.endswith('.jpg')])
            if existing_samples >= samples_per_class:
                print(f"Already have {existing_samples} samples for {word}. Skipping...")
                continue

            print(f"\nPrepare to collect samples for: {word}")
            print("Press 's' when ready, 'q' to quit")

            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                cv2.putText(frame, f"Collecting: {word}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Press 's' when ready", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Data Collection', frame)

                key = cv2.waitKey(1)
                if key == ord('s'):
                    break
                elif key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            print(f"Collecting {samples_per_class} samples for {word}...")
            count = existing_samples
            while count < samples_per_class:
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)

                if results.multi_hand_landmarks:
                    cv2.imwrite(os.path.join(word_dir, f"{count}.jpg"), frame)
                    count += 1
                    cv2.putText(frame, f"Saved: {count}/{samples_per_class}", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow('Data Collection', frame)
                if cv2.waitKey(50) == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()
        print("\nData collection complete!")

    def train_model(self):
        print("Processing collected data...")

        for word in TARGET_WORDS.values():
            word_dir = os.path.join(self.data_dir, word)
            if not os.path.exists(word_dir):
                continue

            for img_file in os.listdir(word_dir):
                if not img_file.endswith('.jpg'):
                    continue

                img_path = os.path.join(word_dir, img_file)
                landmarks = self._extract_landmarks(img_path)

                if landmarks is not None:
                    self.data.append(landmarks)
                    self.labels.append(word)

        if not self.data:
            print("No valid data found!")
            return

        X = np.array(self.data)
        y = np.array(self.labels)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel trained with accuracy: {accuracy:.2f}")
        print("Sample predictions:", y_pred[:5])

        with open(self.model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {self.model_path}")

    def _extract_landmarks(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            return None

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            return None

        hand_landmarks = results.multi_hand_landmarks[0]
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

        return landmarks
    



if __name__ == "__main__":
    trainer = SimpleSignLanguageTrainer()
    trainer.collect_samples(samples_per_class=50)
    trainer.train_model()
