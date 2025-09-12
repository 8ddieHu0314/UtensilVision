import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from AVFoundation import AVCaptureDevice, AVMediaTypeVideo
import time

st.title("Real-Time YOLOv8 Detection")

if "run" not in st.session_state:
    st.session_state.run = False

model_selection = st.selectbox(
    'Choose a model:',
    ['yolov8n', 'yolo11n']
)

def list_cameras():
    devices = AVCaptureDevice.devicesWithMediaType_(AVMediaTypeVideo)
    cams = []
    increment = 1

    for d in devices:
        # You can query more properties if needed
        name = d.localizedName()

        cams.append(f"[{increment}] {name}")

        increment += 1
    return cams

cameras = list_cameras()

camera_selection = st.selectbox(
    'Choose a camera',
    cameras
)

# Run and Stop buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("Run"):
        st.session_state.run = True

with col2:
    if st.button("Stop"):
        st.session_state.run = False

if  st.session_state.run and camera_selection and model_selection:
    model = YOLO(model_selection)

    camera_index = cameras.index(camera_selection)

    # Start video capture
    cap = cv2.VideoCapture(camera_index)

    # Streamlit placeholder for video
    frame_window = st.image([])

    while st.session_state.run and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to grab frame")
            break

        if camera_index == 0: frame = cv2.flip(frame, 1)

        # Run YOLO detection
        results = model.predict(frame, imgsz=320, conf=0.5)

        # Annotate frame
        annotated_frame = results[0].plot()

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Show frame
        frame_window.image(frame_rgb)

        # Add small delay to prevent UI freeze
        time.sleep(0.03)

    cap.release()
    st.write("Camera stopped.")