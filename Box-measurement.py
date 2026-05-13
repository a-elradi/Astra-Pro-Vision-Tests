

import cv2
import numpy as np
from openni import openni2


openni2.initialize(r"C:\OpenNI2\Bin")

dev = openni2.Device.open_any()

depth_stream = dev.create_depth_stream()
depth_stream.start()


cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


FOCAL_LENGTH_PX = 570.0  # approximate for 640x480


def get_depth_at(depth_image, x, y, region=10):
    """
    FIX: Average depth over region x region window.
    Filters invalid (0), below minimum (600mm), above maximum (8000mm).
    """
    h, w = depth_image.shape
    x1 = max(0, x - region)
    x2 = min(w, x + region)
    y1 = max(0, y - region)
    y2 = min(h, y + region)

    patch = depth_image[y1:y2, x1:x2].astype(np.float32)
    valid = patch[(patch >= 600) & (patch <= 8000)]

    if len(valid) == 0:
        return 0, 0  # distance, confidence

    confidence = int((len(valid) / patch.size) * 100)
    return int(np.mean(valid)), confidence

def estimate_real_size(pixel_size, distance_mm):
    """
    Convert pixel width/height to real-world mm using pinhole camera model.
    real_size = (pixel_size * distance) / focal_length
    """
    if distance_mm == 0:
        return 0
    return int((pixel_size * distance_mm) / FOCAL_LENGTH_PX)


while True:

    ret, frame = cap.read()
    if not ret:
        break

    # Depth frame
    depth_frame = depth_stream.read_frame()
    depth_data = depth_frame.get_buffer_as_uint16()
    depth_image = np.frombuffer(depth_data, dtype=np.uint16).copy()
    depth_image = depth_image.reshape((480, 640))

    # ── BOX DETECTION 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)  

    # FIX: Dilate edges to close small gaps in box outline
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    best_box = None
    best_area = 0

    for cnt in contours:

        area = cv2.contourArea(cnt)
        if area < 5000:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

        
        if not (4 <= len(approx) <= 6):
            continue

        x, y, w, h = cv2.boundingRect(approx)

      
        aspect = w / h if h > 0 else 0
        if aspect < 0.3 or aspect > 3.5:
            continue

        if area > best_area:
            best_area = area
            best_box = (x, y, w, h, approx)

    # ── DRAW AND MEASURE 
    if best_box:

        x, y, w, h, approx = best_box

        cx = x + w // 2
        cy = y + h // 2

        
        dx = int(cx * 640 / frame.shape[1])
        dy = int(cy * 480 / frame.shape[0])
        dx = np.clip(dx, 0, 639)
        dy = np.clip(dy, 0, 479)

        
        distance_mm, confidence = get_depth_at(depth_image, dx, dy)

        
        real_w_mm = estimate_real_size(w, distance_mm)
        real_h_mm = estimate_real_size(h, distance_mm)

        # Draw
        cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # Distance label
        if distance_mm > 0:
            dist_color = (0, 255, 0) if confidence > 50 else (0, 165, 255)
            cv2.putText(frame,
                        f"Dist: {distance_mm}mm ({distance_mm/10:.1f}cm) [{confidence}%]",
                        (x, y - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, dist_color, 2)
            cv2.putText(frame,
                        f"Real: {real_w_mm}mm x {real_h_mm}mm",
                        (x, y - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)
        else:
            cv2.putText(frame, "Distance: OUT OF RANGE",
                        (x, y - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        # Pixel size label
        cv2.putText(frame, f"Px: {w}x{h}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

    cv2.imshow("Box Measurement", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
depth_stream.stop()
openni2.unload()
cv2.destroyAllWindows()