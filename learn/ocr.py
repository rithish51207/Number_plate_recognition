import cv2
from easyocr import Reader
reader= Reader(['en'])


threshold=0.80
def read_plate(plate_image):
    
    # Step 1: Resize
    plate_image = cv2.resize(
        plate_image,
        None,
        fx=3,
        fy=3,
        interpolation= cv2.INTER_CUBIC
    )

    # Step 2: Convert to grayscale
    plate_image = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

    plate_image = clahe.apply(plate_image)
    results_ocr=[]
    results=reader.readtext(plate_image)
    for result in results:
        txt_con=result[2]
        txt=result[1]
        if txt_con >= threshold:
           results_ocr.append({
           "text": txt,
            "confidence": txt_con
        })
    return results_ocr