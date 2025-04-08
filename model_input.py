import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import numpy as np
import cv2
import pygetwindow as gw
import argparse
import mss


# Command format: python model_input.py path/to/EfficientNetV2/model (camera|window) [--window_name "window name"] [--camera_index camera_index]
# Camera example: python model_input.py path/to/model camera --camera_index 1
# Window example: python model_input.py path/to/model window --window_name "Media Player"
# Window names will be displayed if window is not found
# Camera index: webcam is typically 0, additional cameras will be 1, 2, etc.
# To use different model type, edit preprocess_frame and extract_prediction methods


# Preprocesses frame for EfficientNetV2 model
def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))
    frame_array = np.array(frame, dtype=np.float32)
    frame_array = preprocess_input(frame_array)
    input_data = np.expand_dims(frame_array, axis=0)
    return input_data

# Extract prediction value
def extract_prediction(pred):
    predicted_tensor = pred['dense_5']
    return predicted_tensor.numpy().item()


# Capture and process frames from camera input
def camera(model, camera_input):
    # Initialize camera
    cap = cv2.VideoCapture(camera_input)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Process frame
        input_data = preprocess_frame(frame)
        # Run model on frame
        prediction = model(input_data)
        # Extract prediction value, 0 is solid, 1 is transparent
        predicted_value = extract_prediction(prediction)
        # Prepare prediction text
        prediction_text = "Prediction: Transparent" if predicted_value >= 0.5 else "Prediction: Solid"
        prediction_text += f" ({predicted_value:.2f})"
        # Display image with prediction
        cv2.putText(frame, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Live Video Feed', frame)


        # Stop running with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Capture and process frames from window input
def window(model, window_name):
    # Find window
    windows = gw.getWindowsWithTitle(window_name)
    if not windows:
        # Show open windows if not found
        print(f"No window found with the title: '{window_name}'")
        print("Windows currently open:")
        window_titles = gw.getAllTitles()
        for title in window_titles:
            print(title)
    else:
        window = windows[0]
        with mss.mss() as sct:
            while True:
                # Locate window
                x, y, w, h = window.left, window.top, window.width, window.height
                monitor = {"top": y, "left": x, "width": w, "height": h}
                # Capture the screen content of the window
                sct_img = sct.grab(monitor)
                # Convert captured image
                frame = np.array(sct_img)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                # Process frame
                input_data = preprocess_frame(frame)
                # Run model on frame
                prediction = model(input_data)
                # Extract prediction value, 0 is solid, 1 is transparent
                predicted_value = extract_prediction(prediction)
                # Prepare prediction text
                prediction_text = "Prediction: Transparent" if predicted_value >= 0.5 else "Prediction: Solid"
                prediction_text += f" ({predicted_value:.2f})"
                # Display image with prediction
                cv2.putText(frame, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("Captured Window", frame)

                # Stop running with 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # Create parser
    parser = argparse.ArgumentParser(description="Runs model on image from camera or window")
    parser.add_argument("model", type=str, help="Path to EfficientNetV2 model")
    parser.add_argument("capture", type=str, help="Capture from camera or window")
    parser.add_argument("--window_name", type=str, help="Name of window to capture", default="")
    parser.add_argument("--camera_input", type=int, help="Input channel of camera", default=0)
    args = parser.parse_args()
    model = tf.keras.layers.TFSMLayer(args.model, call_endpoint='serving_default')
    if args.capture == "camera":
        camera(model, args.camera_input)
    elif args.capture == "window":
        window(model, args.window_name)