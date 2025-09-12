import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import numpy as np

st.title("Real-Time YOLOv8 Detection with Webcam")

# Load YOLO model
model = YOLO("yolo11n.pt")  # Replace with your custom model if needed

# Start video capture
cap = cv2.VideoCapture(0)

# Streamlit placeholder for video
frame_window = st.image([])

# Stop button
stop_button = st.button("Stop")

while cap.isOpened() and not stop_button:
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)

    # Run YOLO inference
    results = model.predict(frame, imgsz=320, conf=0.5)

    # Annotate frame with detections
    annotated_frame = results[0].plot()

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    # Display frame in Streamlit
    frame_window.image(frame_rgb)

cap.release()
st.write("Webcam stopped.")