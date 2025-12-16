import cv2
import os
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from flask import Flask, render_template, Response



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('Alphabets')

# Actions that we try to detect
actions = []

# 30 videos worth of data
no_sequence = 30

# Videos are going to be 30 frames in length
sequence_length = 30

for root, dirs, files in os.walk(DATA_PATH):
    for dir_name in dirs:
        if not dir_name.isdigit():
            actions.append(dir_name)

actions = np.array(actions)
print(actions)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 126)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.load_weights('alphabets.h5')


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_styled_landmarks(image, results):
    # Draw left hand connections
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Determine if it's the left hand or right hand based on landmark positions
            if landmarks.landmark[mp_hands.HandLandmark.WRIST].x < landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x:
                # Left hand
                mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS,
                                           mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                           mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                           )
            else:
                # Right hand
                mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS,
                                           mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                           mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                           )


def extract_keypoints(results):
    # Initialize empty arrays for left and right hand landmarks
    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)

    # Check if multi_hand_landmarks are available
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Determine if it's the left hand or right hand based on landmark positions
            if landmarks.landmark[mp_hands.HandLandmark.WRIST].x < landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x:
                # Left hand
                lh = np.array([[res.x, res.y, res.z] for res in landmarks.landmark]).flatten()
            else:
                # Right hand
                rh = np.array([[res.x, res.y, res.z] for res in landmarks.landmark]).flatten()

    return np.concatenate([lh, rh])

# 1. New detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.7

cap = cv2.VideoCapture(0)

# Set mediapipe model 
def generate_frames():
    global sequence
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))
                
                if np.unique(predictions[-10:])[0] == np.argmax(res) and res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])
                
                if len(sentence) > 5:
                    sentence = sentence[-5:]
            
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
