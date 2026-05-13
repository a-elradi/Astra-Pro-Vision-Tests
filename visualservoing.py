"""
visualservoing.py — CORRECTED
Fixes:
- Region-averaged depth (10x10 patch) instead of single pixel
- Invalid depth filtering (0, <600mm, >8000mm removed)
- Removed CAP_DSHOW (Windows-only, crashes on Jetson)
- Relaxed contour detection (4-6 corners)
- Aspect ratio filter added
- Depth stability buffer — average last N readings for smooth lift height
- Better alignment threshold with hysteresis (enter at 10px, exit at 20px)
- Added visual center line for easier alignment visualization
- Added lift height output clearly displayed
- Proportional correction scaled properly
"""

from openni import openni2
import numpy as np
import cv2
from collections import deque

# ── INIT ─────────────────────────────────────────────────────────────────────
openni2.initialize(r"C:\OpenNI2\Bin")

dev = openni2.Device.open_any()
depth_stream = dev.create_depth_stream()
depth_stream.start()

# FIX: Removed CAP_DSHOW — works on both Windows and Jetson
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# FIX: Depth stability buffer — smooth out noisy readings
DEPTH_BUFFER_SIZE = 10
depth_buffer = deque(maxlen=DEPTH_BUFFER_SIZE)

# Alignment state (hysteresis prevents flickering)
ALIGN_ENTER_THRESHOLD = 10   # px — enter aligned state
ALIGN_EXIT_THRESHOLD  = 20   # px — exit aligned state
is_aligned = False

# ── HELPER: region-averaged depth ────────────────────────────────────────────
def get_depth_at(depth_image, x, y, region=10):
    """
    FIX: Average over region. Filter invalid and out-of-range values.
    """
    h, w = depth_image.shape
    x1 = max(0, x - region)
    x2 = min(w, x + region)
    y1 = max(0, y - region)
    y2 = min(h, y + region)

    patch = depth_image[y1:y2, x1:x2].astype(np.float32)
    valid = patch[(patch >= 600) & (patch <= 8000)]

    if len(valid) == 0:
        return 0

    return int(np.mean(valid))

# ── HELPER: stable lift height from buffer ───────────────────────────────────
def get_stable_depth(raw_depth):
    """
    FIX: Add to rolling buffer and return median.
    Median is better than mean for depth — rejects spike outliers.
    """
    if raw_depth > 0:
        depth_buffer.append(raw_depth)

    if len(depth_buffer) == 0:
        return 0

    return int(np.median(depth_buffer))

# ── HELPER: visual servo correction ──────────────────────────────────────────
def visual_servo(box_cx, frame_width, Kp=0.008):
    """
    FIX: Kp adjusted (0.005 was too small, caused sluggish response).
    Added hysteresis: separate enter/exit thresholds.
    Returns:
        correction — float, positive=right, negative=left
        offset_px  — raw pixel offset
        aligned    — bool
    """
    global is_aligned

    frame_center = frame_width / 2
    offset = box_cx - frame_center
    correction = float(offset) * Kp

    # Hysteresis logic
    if is_aligned:
        # Already aligned — only exit if offset gets large
        if abs(offset) > ALIGN_EXIT_THRESHOLD:
            is_aligned = False
    else:
        # Not aligned — only enter if offset gets small
        if abs(offset) < ALIGN_ENTER_THRESHOLD:
            is_aligned = True

    if is_aligned:
        correction = 0.0

    return correction, int(offset), is_aligned

# ── HELPER: box detection ────────────────────────────────────────────────────
def detect_box(rgb_frame, depth_image):
    """
    FIX: Accept 4-6 corner contours (not exactly 4).
    FIX: Added aspect ratio filter.
    FIX: Use region-averaged depth.
    """
    gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)

    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    best = None
    best_area = 0

    for cnt in contours:

        area = cv2.contourArea(cnt)
        if area < 8000:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

        # FIX: Accept 4 to 6 corners
        if not (4 <= len(approx) <= 6):
            continue

        x, y, w, h = cv2.boundingRect(approx)

        # FIX: Aspect ratio filter
        aspect = w / h if h > 0 else 0
        if aspect < 0.3 or aspect > 3.5:
            continue

        if area > best_area:
            best_area = area
            best = (x, y, w, h)

    if best:
        x, y, w, h = best
        cx = x + w // 2
        cy = y + h // 2

        dx = np.clip(int(cx * 640 / rgb_frame.shape[1]), 0, 639)
        dy = np.clip(int(cy * 480 / rgb_frame.shape[0]), 0, 479)

        # FIX: Region-averaged depth
        raw_depth = get_depth_at(depth_image, dx, dy)

        # FIX: Stable depth through buffer
        stable_depth = get_stable_depth(raw_depth)

        return x, y, w, h, cx, cy, stable_depth

    return None

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
print("Visual Servo started — Press Q to quit")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    depth_frame = depth_stream.read_frame()
    depth_data = depth_frame.get_buffer_as_uint16()
    depth_image = np.frombuffer(depth_data, dtype=np.uint16).copy()
    depth_image = depth_image.reshape((480, 640))

    # FIX: Draw center line for visual alignment reference
    frame_w = frame.shape[1]
    frame_h = frame.shape[0]
    cv2.line(frame,
             (frame_w // 2, 0),
             (frame_w // 2, frame_h),
             (200, 200, 200), 1)

    result = detect_box(frame, depth_image)

    if result:

        x, y, w, h, cx, cy, stable_depth = result

        # ── VISUAL SERVO ─────────────────────────────────────────────────
        correction, offset_px, aligned = visual_servo(cx, frame_w)

        # ── DRAW ─────────────────────────────────────────────────────────
        box_color = (0, 255, 0) if aligned else (0, 165, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 3)
        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

        # Line from box center to frame center
        cv2.line(frame,
                 (cx, cy),
                 (frame_w // 2, cy),
                 (255, 0, 255), 2)

        # ── STATUS PANEL ─────────────────────────────────────────────────
        if aligned:
            status = "ALIGNED"
            status_color = (0, 255, 0)
        elif correction > 0:
            status = "MOVE RIGHT"
            status_color = (0, 165, 255)
        else:
            status = "MOVE LEFT"
            status_color = (0, 165, 255)

        # Background panel
        cv2.rectangle(frame, (0, 0), (400, 120), (30, 30, 30), -1)

        cv2.putText(frame, status,
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.1, status_color, 2)

        cv2.putText(frame, f"Offset: {offset_px}px",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.putText(frame, f"Correction: {correction:.4f}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # FIX: Lift height clearly displayed
        if stable_depth > 0:
            cv2.putText(frame,
                        f"Lift Height: {stable_depth}mm ({stable_depth/10:.1f}cm)",
                        (10, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "Lift Height: OUT OF RANGE",
                        (10, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Distance on box
        if stable_depth > 0:
            cv2.putText(frame,
                        f"{stable_depth}mm",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Console
        print(f"{status:12} | Offset: {offset_px:+4d}px "
              f"| Correction: {correction:+.4f} "
              f"| Depth: {stable_depth}mm")

    else:
        cv2.putText(frame, "No box detected",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        # FIX: Clear buffer when box lost
        depth_buffer.clear()
        is_aligned = False

    cv2.imshow("Visual Servoing", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
depth_stream.stop()
openni2.unload()
cv2.destroyAllWindows()