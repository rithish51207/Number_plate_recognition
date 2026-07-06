import cv2
import time

from detector import detect_plates
from ocr import read_plate
from logger import log_plate

COOLDOWN = 10
plate_cache = {}

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    detections = detect_plates(frame)

    for detection in detections:

        x1 = detection["x1"]
        y1 = detection["y1"]
        x2 = detection["x2"]
        y2 = detection["y2"]

        plate = frame[y1:y2, x1:x2]

        result_ocr = read_plate(plate)

        if not result_ocr:
            continue

        for tex in result_ocr:

            plate_text = tex["text"]
            confidence = tex["confidence"]

            image_name = "plate.jpg"

            current_time = time.time()

            if plate_text in plate_cache:

                last_seen = plate_cache[plate_text]

                if current_time >= last_seen + COOLDOWN:

                    log_plate(plate_text, confidence, image_name)
                    plate_cache[plate_text] = current_time

            else:

                log_plate(plate_text, confidence, image_name)
                plate_cache[plate_text] = current_time

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                plate_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.78,
                (0, 0, 255),
                3
            )

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()