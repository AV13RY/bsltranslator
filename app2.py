from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import pickle

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# Load the pre-trained SVM models
try:
    with open("one_hand_model.pkl", "rb") as f:
        one_hand_model = pickle.load(f)

    with open("two_hand_model.pkl", "rb") as f:
        two_hand_model = pickle.load(f)
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure the .pkl files are in the correct location.")
    exit()

# Define class labels (mapping from model output to gesture names)
class_labels = {
    "0 - 0": "0",
    "1 - 1": "1",
    "2 - 2": "2",
    "3 - 3": "3",
    "4 - 4": "4",
    "5 - 5": "5",
    "6 - 6": "6",
    "7 - 7": "7",
    "8 - 8": "8",
    "9 - 9": "9",
    "10 - 10": "10",
    "A - a": "A",
    "B - b": "B",
    "C - c": "C",
    "D - d": "D",
    "E - e": "E",
    "F - f": "F",
    "G - g": "G",
    "I - i": "I",
    "K - k": "K",
    "L - l": "L",
    "M - m": "M",
    "N - n": "N",
    "O - o": "O",
    "P - p": "P",
    "Q - q": "Q",
    "R - r": "R",
    "S - s": "S",
    "T - t": "T",
    "U - u": "U",
    "V - v": "V",
    "W - w": "W",
    "X - x": "X",
    "Z - z": "Z"
}

# Function to extract hand landmarks using MediaPipe
def extract_hand_landmarks(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    landmarks = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract 21 hand landmarks (x, y, z coordinates)
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
    return landmarks

# Function to preprocess landmarks for SVM input
def preprocess_landmarks(landmarks):
    # Convert to numpy array and reshape for SVM input
    return np.array(landmarks).reshape(1, -1)

@app.route('/detect-gesture', methods=['POST'])
def detect_gesture():
    # Get the image from the request
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Extract hand landmarks
    landmarks = extract_hand_landmarks(image)

    if landmarks:
        # Preprocess landmarks for SVM input
        input_data = preprocess_landmarks(landmarks)

        # Determine if it's a one-handed or two-handed gesture
        if len(landmarks) == 63:  # 21 landmarks * 3 coordinates (x, y, z)
            # One-handed gesture
            prediction = one_hand_model.predict(input_data)
            gesture = class_labels.get(prediction[0], "Unknown Gesture")
            return jsonify({'gesture': gesture, 'type': 'One-Hand Gesture'})
        elif len(landmarks) == 126:  # 42 landmarks * 3 coordinates (x, y, z)
            # Two-handed gesture
            prediction = two_hand_model.predict(input_data)
            gesture = class_labels.get(prediction[0], "Unknown Gesture")
            return jsonify({'gesture': gesture, 'type': 'Two-Hand Gesture'})
        else:
            # Invalid number of landmarks
            return jsonify({'gesture': 'Invalid Gesture', 'type': 'Error'})
    else:
        return jsonify({'gesture': 'No hand detected', 'type': 'Error'})

if __name__ == '__main__':
    app.run(debug=True)