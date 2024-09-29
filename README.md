# Hand-Volume-Controller

## Hand Gesture Volume Control

This project implements a hand gesture recognition system to control the system volume using a webcam. The system uses MediaPipe for hand landmark detection and a Random Forest classifier to predict hand gestures. The predicted gestures are then used to adjust the system volume gradually.

## Features

- **Hand Landmark Detection**: Utilizes MediaPipe to detect hand landmarks in real-time.
- **Gesture Recognition**: Trained a Random Forest classifier to recognize specific hand gestures.
- **Volume Control**: Adjusts the system volume based on the recognized hand gestures.
- **Real-time Processing**: Processes video frames in real-time to provide immediate feedback.

## Project Structure

- `create_images.py`: Captures images from the webcam to create a dataset for training the gesture recognition model.
- `create_dataset.py`: Processes the captured images to extract hand landmarks and create a dataset for training.
- `random_forest_model.py`: Trains a Random Forest classifier using the processed dataset and saves the trained model.
- `volume_controller.py`: Contains functions to control the system volume using the Pycaw library.
- `hand_detector.py`: Main script that captures video from the webcam, detects hand landmarks, predicts gestures, and adjusts the system volume accordingly.

## How It Works

1. **Data Collection**: Use `create_images.py` to capture images for different hand gestures.
2. **Dataset Creation**: Run `create_dataset.py` to process the captured images and create a dataset of hand landmarks.
3. **Model Training**: Use `random_forest_model.py` to train a Random Forest classifier on the created dataset.
4. **Volume Control**: Execute `hand_detector.py` to start the real-time hand gesture recognition and volume control.

## Requirements

- Python 3.x
- OpenCV
- MediaPipe
- Scikit-learn
- Matplotlib
- Pycaw
- Comtypes

## Installation

### Clone the repository:
   ```bash
   git clone https://github.com/yourusername/hand-volume-controller.git
   ```

## Usage
1. Collect Images:
   ```bash
   python create_images.py
   ```

2. Create Dataset:
   ```bash
   python create_dataset.py
   ```
3. Train Model:
   ```bash
   python random_forest_model.py
   ```
4. Run Hand Detector:
   ```bash
   python hand_detector.py
   ```
