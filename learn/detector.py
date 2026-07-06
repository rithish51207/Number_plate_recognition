from ultralytics import YOLO
model=YOLO("best.pt")

def detect_plates(frame):

    results = model(frame)

    detections = []

    for result in results:

        for box in result.boxes:
            confidence =float (box.conf[0])
            class_id = int(box.cls[0])
            if confidence >= 0.80 and class_id== 0:

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detection = {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": confidence,
                "class_id": class_id
                }
                detections.append(detection)
    return detections