import supervision as sv

def run_detection(model, frame):
    class_names = {0: "Fork", 1: "Knife", 2: "Spoon"}

    results = model.infer(
        frame, 
        imgzs=640, 
        confidence=0.65, 
        iou_threshold=0.7
    )[0]

    detections = sv.Detections.from_inference(results)

    labels = [class_names.get(cls, str(cls)) for cls in detections.class_id]

    fork_count = len(detections[detections.class_id == 0])
    knife_count = len(detections[detections.class_id == 1])
    spoon_count = len(detections[detections.class_id == 2])

    return fork_count, knife_count, spoon_count, labels, detections