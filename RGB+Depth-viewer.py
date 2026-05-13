"""
RGB+Depth-viewer.py — CORRECTED
Fixes:
- Added colormap for depth display
- Added center distance with region averaging + invalid filtering
- Added side-by-side display for easy comparison
- Fixed camera index comment to help identify correct index
"""

from openni import openni2
import numpy as np
import cv2

# ── INIT ─────────────────────────────────────────────────────────────────────
openni2.initialize(r"C:\OpenNI2\Bin")

dev = openni2.Device.open_any()

depth_stream = dev.create_depth_stream()
depth_stream.start()

# FIX: Try index 0 first — if black screen try index 1
# Astra Pro RGB sometimes appears as a separate device
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Index 0 failed — trying index 1")
    cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Cannot open RGB camera — check USB connection")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("RGB + Depth started — Press Q to quit")

# ── HELPER ───────────────────────────────────────────────────────────────────
def get_depth_at(depth_image, x, y, region=10):
    h, w = depth_image.shape
    x1, x2 = max(0, x - region), min(w, x + region)
    y1, y2 = max(0, y - region), min(h, y + region)
    patch = depth_image[y1:y2, x1:x2].astype(np.float32)
    valid = patch[(patch >= 600) & (patch <= 8000)]
    return int(np.mean(valid)) if len(valid) > 0 else 0

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
while True:

    ret, color_frame = cap.read()
    if not ret:
        print("Failed to read RGB frame")
        break

    depth_frame = depth_stream.read_frame()
    depth_data = depth_frame.get_buffer_as_uint16()

    depth_array = np.frombuffer(depth_data, dtype=np.uint16).copy()
    depth_array = depth_array.reshape((480, 640))

    # FIX: Normalize clipped range + colormap
    display = depth_array.astype(np.float32)
    display = np.clip(display, 600, 4000)
    display = cv2.normalize(display, None, 0, 255, cv2.NORM_MINMAX)
    depth_colormap = cv2.applyColorMap(np.uint8(display), cv2.COLORMAP_JET)

    # Resize color to match depth if needed
    color_frame = cv2.resize(color_frame, (640, 480))

    # FIX: Show center distance on both frames
    cx, cy = 320, 240
    dist = get_depth_at(depth_array, cx, cy)

    for img in [color_frame, depth_colormap]:
        cv2.drawMarker(img, (cx, cy), (255, 255, 255),
                       cv2.MARKER_CROSS, 20, 2)
        label = f"{dist}mm" if dist > 0 else "OUT OF RANGE"
        cv2.putText(img, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # FIX: Side-by-side display
    combined = np.hstack([color_frame, depth_colormap])
    cv2.imshow("RGB  |  Depth", combined)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
depth_stream.stop()
openni2.unload()
cv2.destroyAllWindows()