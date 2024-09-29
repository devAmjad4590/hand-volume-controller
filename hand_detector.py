import cv2
import mediapipe as mp
import pickle
import numpy as np
from volume_controller import get_system_volume_controller, reduce_volume_gradually, getPrint
import threading
import time
import warnings


warnings.filterwarnings("ignore", category=UserWarning, message="SymbolDatabase.GetPrototype() is deprecated. Please use message_factory.GetMessageClass() instead. SymbolDatabase.GetPrototype() will be removed soon.")

# Initialize MediaPipe Hand Detector
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mphands = mp.solutions.hands

# Load the trained model
with open('model.p', 'rb') as f:
    model_dict = pickle.load(f)
    model = model_dict['model']

# Global variable to control prediction delay
last_prediction_time = 0
prediction_interval = 0.1  # seconds

def get_hand_landmarks():
    camera = cv2.VideoCapture(0)
    hands = mphands.Hands(static_image_mode=True, min_detection_confidence=0.3)
    volume_controller = get_system_volume_controller()  # Get the volume controller instance

    def delayed_prediction(data_aux):
        global last_prediction_time
        current_time = time.time()
        if current_time - last_prediction_time >= prediction_interval:
            last_prediction_time = current_time
            prediction = model.predict([np.asarray(data_aux)])
            is_up = prediction[0] == '1'
            reduce_volume_gradually(volume_controller, target_volume=0.0 if not is_up else 1.0, step=0.05, up=is_up)

    while camera.isOpened():
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = camera.read()

        H, W, _ = frame.shape
        if not ret:
            continue
        # Convert the BGR image to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)
        results = hands.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Check if the detected hand is the right hand
                handedness = results.multi_handedness[idx].classification[0].label
                if handedness == 'Right':
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mphands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                    # Extract the hand landmarks coordinates
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                    # Start a new thread for delayed prediction
                    threading.Thread(target=delayed_prediction, args=(data_aux,)).start()
        label = f"Volume: {getPrint() * 100:.0f}%"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Hand Landmarks', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

get_hand_landmarks()