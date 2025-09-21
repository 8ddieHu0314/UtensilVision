import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from AVFoundation import AVCaptureDevice, AVMediaTypeVideo
from inference import get_model
import supervision as sv
import random
import pandas as pd

def generate_sample_data(dining_halls):
    data = {}
    sessions = ['Breakfast', 'Lunch', 'Dinner']
    utensils = ['Forks', 'Knives', 'Spoons']

    for hall in dining_halls:
        hall_data = {}
        for session in sessions:
            session_data = {utensil: random.randint(50, 300) for utensil in utensils}
            hall_data[session] = session_data
        data[hall] = hall_data

    return data

dining_halls = [
    'Cook House Dining Room',
    'Becker House Dining Room',
    'North Star Dining Room',
    'Okenshields',
    'Risley Dining',
    'Morrison Dining',
    'Jansen\'s at Bethe House',
    '104West!'
]

utensils_data = generate_sample_data(dining_halls)

def image_detection_page():
    st.title("Real-Time YOLOv11 Detection")

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

        box_annotator = sv.BoxAnnotator(thickness=3)
        
        label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1.2)

        # Class names mapping
        class_names = {0: "Fork", 1: "Knife", 2: "Spoon"}

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
                iou_threshold=0.4
            )[0]

            detections = sv.Detections.from_inference(results)

            labels = [class_names.get(cls, str(cls)) for cls in detections.class_id]

            # Count objects by class
            fork_count = len(detections[detections.class_id == 0])
            knife_count = len(detections[detections.class_id == 1])
            spoon_count = len(detections[detections.class_id == 2])

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


def data_analytics_page():
    st.title('Cornell Dining Halls Utensils Tracker')
    st.write(
        'This app displays the individual utensil counts (Forks, Knives, Spoons) '
        'for each meal session in different dining halls at Cornell. '
        'Select a dining hall and a meal session to view details.'
    )

    tab_titles = dining_halls
    tabs = st.tabs(tab_titles)

    for i, tab in enumerate(tabs):
        with tab:
            hall = dining_halls[i]
            st.subheader(f'{hall} - Select a Meal Session')

            col1, col2, col3 = st.columns(3)
            selected_session = None

            with col1:
                if st.button("Breakfast", key=f"{hall}-breakfast"):
                    selected_session = "Breakfast"
            with col2:
                if st.button("Lunch", key=f"{hall}-lunch"):
                    selected_session = "Lunch"
            with col3:
                if st.button("Dinner", key=f"{hall}-dinner"):
                    selected_session = "Dinner"

            if selected_session:
                st.markdown(f"### {selected_session} Utensil Counts")

                try:
                    session_data = utensils_data[hall][selected_session]
                    if isinstance(session_data, dict) and all(k in session_data for k in ['Forks', 'Knives', 'Spoons']):
                        df = pd.DataFrame({
                            'Utensil': ['Forks', 'Knives', 'Spoons'],
                            'Count': [
                                session_data['Forks'],
                                session_data['Knives'],
                                session_data['Spoons']
                            ]
                        }).set_index('Utensil')

                        st.table(df)
                        st.subheader("Histogram Visualization")
                        st.bar_chart(df)
                    else:
                        st.error(f"Data for {hall} - {selected_session} is not in the expected format.")
                except KeyError:
                    st.error(f"No data available for {hall} - {selected_session}.")

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Data Analytics", "Image Detection"])

    if page == "Data Analytics":
        data_analytics_page()
    elif page == "Image Detection":
        image_detection_page()

if __name__ == "__main__":
    main()