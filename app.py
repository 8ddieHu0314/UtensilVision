import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from AVFoundation import AVCaptureDevice, AVMediaTypeVideo
from inference import get_model
import supervision as sv
import cv2


st.title("Real-Time YOLOv8 Detection")

if "run" not in st.session_state:
    st.session_state.run = False

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

if  st.session_state.run and camera_selection:
    model = get_model(model_id="utensils-jabsv/2")

    camera_index = cameras.index(camera_selection)

    # Start video capture
    cap = cv2.VideoCapture(camera_index)

    # Streamlit placeholder for video
    frame_window = st.image([])

    box_annotator = sv.BoxAnnotator(
        thickness=3,
    )
    
    label_annotator = sv.LabelAnnotator(
        text_thickness=2,
        text_scale=1.2
    )

    while st.session_state.run and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to grab frame")
            break

        if camera_index == 0: frame = cv2.flip(frame, 1)

        # Run YOLO detection
        results = model.infer(
            frame, 
            imgzs=640, 
            confidence=0.6, 
            iou_threshold=0.75
        )[0]

        detections = sv.Detections.from_inference(results)

        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

        # Convert BGR to RGB for display in Streamlit
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Show the frame
        frame_window.image(frame_rgb)

    cap.release()
    st.write("Camera stopped.")