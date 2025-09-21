import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from AVFoundation import AVCaptureDevice, AVMediaTypeVideo
from inference import get_model
import supervision as sv
import cv2
from video_writer import VideoWriter


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

model_selection = st.selectbox(
    'Choose a model',
    [2, 4, 5]
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
    model = get_model(model_id="utensils-jabsv/" + str(model_selection))

    camera_index = cameras.index(camera_selection)

    # Start video capture
    cap = cv2.VideoCapture(camera_index)

    frame_window = st.image([])

    count_placeholder = st.empty()

    box_annotator = sv.BoxAnnotator(
        thickness=3,
    )
    
    label_annotator = sv.LabelAnnotator(
        text_thickness=2,
        text_scale=1.2
    )

    # Class names mapping
    class_names = {0: "Fork", 1: "Knife", 2: "Fork"}

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
            confidence=0.5, 
            iou_threshold=0.6
        )[0]

        detections = sv.Detections.from_inference(results)
        detections.class_id = np.where(detections.class_id == 2, 0, detections.class_id)
        detections.class_id = np.where(detections.class_id == 1, 0, detections.class_id)

        labels = [class_names.get(cls, str(cls)) for cls in detections.class_id]

        # Count objects by class
        fork_count = len(detections[detections.class_id == 0])
        knife_count = len(detections[detections.class_id == 0])
        spoon_count = len(detections[detections.class_id == 0])

        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        # Convert BGR to RGB for display in Streamlit
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Show the frame
        frame_window.image(frame_rgb)
        
        # Update count display
        with count_placeholder.container():
            st.markdown("### Utensil Count")
            col_fork, col_knife, col_spoon = st.columns(3)
            
            with col_fork:
                st.metric("Forks", fork_count)
            with col_knife:
                st.metric("Knives", knife_count)
            with col_spoon:
                st.metric("Spoons", spoon_count)

            st.markdown(f"**Total:** {fork_count + knife_count + spoon_count}")

    cap.release()
    st.write("Camera stopped.")