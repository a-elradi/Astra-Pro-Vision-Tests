"""
QR-Scanner-OpenCV.py — CORRECTED
Fixes:
- Added CLAHE contrast enhancement before detection (main accuracy fix)
- Added grayscale preprocessing
- Added pyzbar as fallback when OpenCV QR fails
- Added scan confirmation (must match N times before confirming)
- Fixed resolution to 640x480 (Astra Pro RGB native)
- Added visual feedback with bounding box and data display
"""

import cv2
import numpy as np
from pyzbar import pyzbar  # pip install pyzbar

# ── INIT ─────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)

# FIX: Use native resolution — Astra Pro RGB is 640x480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# FIX: Increase autofocus and exposure
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

detector = cv2.QRCodeDetector()

# FIX: Scan confirmation — require N consistent reads before confirming
CONFIRM_THRESHOLD = 5
scan_count = 0
last_data = ""
confirmed_data = None

print("QR Scanner started — Press Q to quit")

# ── HELPER: enhance frame for QR detection ───────────────────────────────────
def enhance_for_qr(frame):
    """
    Apply CLAHE contrast enhancement and sharpening.
    This is the main fix for poor accuracy in warehouse lighting.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # CLAHE: Contrast Limited Adaptive Histogram Equalization
    # Improves QR detection in uneven lighting dramatically
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Sharpening kernel
    kernel = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    return sharpened

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
while True:

    ret, frame = cap.read()
    if not ret:
        break

    detected_data = None

    # FIX: Try OpenCV detector on enhanced frame first
    enhanced = enhance_for_qr(frame)
    data, bbox, _ = detector.detectAndDecode(enhanced)

    if data:
        detected_data = data

    # FIX: If OpenCV fails, try pyzbar as fallback
    if not detected_data:
        decoded_objects = pyzbar.decode(enhanced)
        if decoded_objects:
            detected_data = decoded_objects[0].data.decode("utf-8")

    # FIX: Confirmation logic — only confirm after N consistent reads
    if detected_data:
        if detected_data == last_data:
            scan_count += 1
        else:
            scan_count = 1
            last_data = detected_data

        if scan_count >= CONFIRM_THRESHOLD:
            confirmed_data = detected_data

        # Draw bounding box if OpenCV found it
        if bbox is not None:
            bbox = bbox.astype(int)
            for i in range(len(bbox[0])):
                pt1 = tuple(bbox[0][i])
                pt2 = tuple(bbox[0][(i + 1) % len(bbox[0])])
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    # ── DISPLAY STATUS ──────────────────────────────────────────────────────
    if confirmed_data:
        # Green confirmed box
        cv2.rectangle(frame, (0, 0), (640, 60), (0, 180, 0), -1)
        cv2.putText(frame, f"CONFIRMED: {confirmed_data}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 2)
        print(f"QR CONFIRMED: {confirmed_data}")

    elif detected_data:
        # Yellow — reading but not confirmed yet
        cv2.rectangle(frame, (0, 0), (640, 60), (0, 180, 255), -1)
        cv2.putText(frame, f"Reading ({scan_count}/{CONFIRM_THRESHOLD}): {detected_data}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 2)
    else:
        cv2.putText(frame, "Scanning for QR...",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (100, 100, 100), 2)

    # Show scan count bar
    if last_data:
        bar_w = int((scan_count / CONFIRM_THRESHOLD) * 200)
        cv2.rectangle(frame, (10, 460), (10 + bar_w, 475), (0, 255, 0), -1)
        cv2.rectangle(frame, (10, 460), (210, 475), (200, 200, 200), 1)

    cv2.imshow("QR Scanner", frame)

    # Press R to reset confirmation
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('r'):
        confirmed_data = None
        last_data = ""
        scan_count = 0
        print("Reset")

cap.release()
cv2.destroyAllWindows()