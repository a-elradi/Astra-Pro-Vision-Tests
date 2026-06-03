from ultralytics import YOLO
from openni import openni2
import numpy as np
import cv2
from collections import deque

openni2.initialize(r"C:\OpenNI2\Bin")
dev = openni2.Device.open_any()
depth_stream = dev.create_depth_stream()
depth_stream.start()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Load YOLOv8 nano — fastest model
model = YOLO("yolov8n.pt")

# COCO class IDs that could be boxes/containers
# 24=handbag, 26=suitcase, 39=bottle, 56=chair, 73=book
# For cardboard boxes: try detecting class 0 (person) is wrong
# Best approach: use all detections and filter by size
BOX_CLASSES = [24, 26, 64, 67, 73, 76]  # suitcase, book, etc

depth_buffer = deque(maxlen=10)

def get_depth_at(depth_image, x, y, region=10):
    h, w = depth_image.shape
    x1 = max(0, x - region)
    x2 = min(w, x + region)
    y1 = max(0, y - region)
    y2 = min(h, y + region)
    patch = depth_image[y1:y2, x1:x2].astype(np.float32)
    valid = patch[(patch >= 600) & (patch <= 8000)]
    return int(np.mean(valid)) if len(valid) > 0 else 0

# Alignment state
is_aligned = False
ALIGN_ENTER = 10
ALIGN_EXIT = 20

def visual_servo(box_cx, frame_width, Kp=0.008):
    global is_aligned
    offset = box_cx - frame_width / 2
    correction = float(offset) * Kp
    if is_aligned:
        if abs(offset) > ALIGN_EXIT:
            is_aligned = False
    else:
        if abs(offset) < ALIGN_ENTER:
            is_aligned = True
    if is_aligned:
        correction = 0.0
    return correction, int(offset), is_aligned

print("YOLOv8 Vision Pipeline — Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    depth_frame = depth_stream.read_frame()
    depth_data = depth_frame.get_buffer_as_uint16()
    depth_image = np.frombuffer(depth_data, dtype=np.uint16).copy()
    depth_image = depth_image.reshape((480, 640))

    # Run YOLO — confidence 0.3 for warehouse (lower = more sensitive)
    results = model(frame, conf=0.3, verbose=False)

    best_box = None
    best_area = 0

    for r in results[0].boxes:
        x1, y1, x2, y2 = map(int, r.xyxy[0])
        w = x2 - x1
        h = y2 - y1
        area = w * h

        # Filter: must be large enough to be our target box
        if area < 8000:
            continue

        if area > best_area:
            best_area = area
            best_box = (x1, y1, x2, y2, float(r.conf[0]))

    # Draw center line
    fw = frame.shape[1]
    fh = frame.shape[0]
    cv2.line(frame, (fw//2, 0), (fw//2, fh), (200, 200, 200), 1)

    if best_box:
        x1, y1, x2, y2, conf = best_box
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        dx = np.clip(int(cx * 640 / fw), 0, 639)
        dy = np.clip(int(cy * 480 / fh), 0, 479)

        raw_depth = get_depth_at(depth_image, dx, dy)
        if raw_depth > 0:
            depth_buffer.append(raw_depth)
        stable_depth = int(np.median(depth_buffer)) if depth_buffer else 0

        correction, offset_px, aligned = visual_servo(cx, fw)

        # Draw
        color = (0, 255, 0) if aligned else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
        cv2.line(frame, (cx, cy), (fw//2, cy), (255, 0, 255), 2)

        status = "ALIGNED" if aligned else ("MOVE RIGHT" if correction > 0 else "MOVE LEFT")

        cv2.rectangle(frame, (0, 0), (420, 130), (30, 30, 30), -1)
        cv2.putText(frame, status, (10, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (0,255,0) if aligned else (0,165,255), 2)
        cv2.putText(frame, f"Conf: {conf:.0%}  Offset: {offset_px}px",
                    (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
        cv2.putText(frame, f"Correction: {correction:.4f}",
                    (10, 93), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
        cv2.putText(frame,
                    f"Lift Height: {stable_depth}mm" if stable_depth > 0 else "Lift: OUT OF RANGE",
                    (10, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.imshow("YOLOv8 Vision", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
depth_stream.stop()
openni2.unload()
cv2.destroyAllWindows()