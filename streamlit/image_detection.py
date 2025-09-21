#from AVFoundation import AVCaptureDevice, AVMediaTypeVideo
from inference import get_model
import cv2
import supervision as sv
import numpy as np

# def list_cameras():
#     devices = AVCaptureDevice.devicesWithMediaType_(AVMediaTypeVideo)
#     cams = []
#     increment = 1

#     for d in devices:
#         name = d.localizedName()
#         cams.append(f"[{increment}] {name}")
#         increment += 1
#     return cams

def run_detection(camera_selection, model_selection, cameras, frame_window, count_placeholder, run_flag):
    model = get_model(model_id="utensils-jabsv/" + str(model_selection))

    camera_index = cameras.index(camera_selection)

    cap = cv2.VideoCapture(camera_index)

    box_annotator = sv.BoxAnnotator(thickness=3)
    
    label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1.2)

    class_names = {0: "Fork", 1: "Knife", 2: "Spoon"}

    while run_flag() and cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        if camera_index == 0: frame = cv2.flip(frame, 1)

        # Run YOLO detection
        results = model.infer(
            frame, 
            imgzs=640, 
            confidence=0.2, 
            iou_threshold=0.9
        )[0]

        detections = sv.Detections.from_inference(results)
        #detections.class_id = np.where(detections.class_id == 2, 0, detections.class_id)
        #detections.class_id = np.where(detections.class_id == 1, 0, detections.class_id)

        labels = [class_names.get(cls, str(cls)) for cls in detections.class_id]

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
            count_placeholder.markdown("### Utensil Count")
            col_fork, col_knife, col_spoon = count_placeholder.columns(3)
            
            with col_fork:
                col_fork.metric("Forks", fork_count)
            with col_knife:
                col_knife.metric("Knives", knife_count)
            with col_spoon:
                col_spoon.metric("Spoons", spoon_count)

            count_placeholder.markdown(f"**Total:** {fork_count + knife_count + spoon_count}")

    cap.release()