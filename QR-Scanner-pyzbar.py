import cv2
import numpy as np
from pyzbar import pyzbar

# Open Astra RGB camera
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Optional resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def scan_qr(frame):

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Slight blur for stability
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    # Decode QR codes
    decoded = pyzbar.decode(gray)

    for obj in decoded:

        # QR data
        data = obj.data.decode("utf-8")

        # QR polygon points
        pts = obj.polygon
        pts = [(p.x, p.y) for p in pts]

        # Draw QR border
        cv2.polylines(
            frame,
            [np.array(pts)],
            True,
            (0,255,0),
            2
        )

        # Put QR text
        cv2.putText(
            frame,
            data,
            (pts[0][0], pts[0][1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

        return data, frame

    return None, frame


print("QR Scanner Started... Press Q to quit")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to read camera")
        break

    qr_data, output = scan_qr(frame)

    if qr_data:
        print("QR Detected:", qr_data)

    cv2.imshow("QR Scanner", output)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()