"""
Depth-stream.py — CORRECTED
Fixes:
- Added colormap for readable depth visualization
- Added center distance display with region averaging
- Added invalid pixel filtering
- Added minimum distance filter (Astra Pro min = 600mm)
"""

from openni import openni2
import numpy as np
import cv2

# Initialize OpenNI2
openni2.initialize(r"C:\OpenNI2\Bin")

dev = openni2.Device.open_any()

depth_stream = dev.create_depth_stream()
depth_stream.start()

print("Depth stream started... Press Q to quit")

# ── HELPER: safe region-averaged depth ──────────────────────────────────────
def get_depth_at(depth_image, x, y, region=10):
    """
    Returns average depth in mm at (x,y) using region x region window.
    Filters out:
      - Zero values (invalid pixels)
      - Values below 600mm (Astra Pro minimum range)
      - Values above 8000mm (Astra Pro maximum range)
    Returns 0 if no valid reading found.
    """
    h, w = depth_image.shape
    x1 = max(0, x - region)
    x2 = min(w, x + region)
    y1 = max(0, y - region)
    y2 = min(h, y + region)

    patch = depth_image[y1:y2, x1:x2].astype(np.float32)

    # Filter invalid values
    valid = patch[(patch >= 600) & (patch <= 8000)]

    if len(valid) == 0:
        return 0

    return int(np.mean(valid))

# ── MAIN LOOP ────────────────────────────────────────────────────────────────
cv2.namedWindow("Depth Colormap", cv2.WINDOW_NORMAL)

while True:

    frame = depth_stream.read_frame()
    frame_data = frame.get_buffer_as_uint16()

    depth_image = np.frombuffer(frame_data, dtype=np.uint16).copy()
    depth_image = depth_image.reshape((480, 640))

    # FIX: Normalize only valid range for better contrast
    display = depth_image.copy().astype(np.float32)
    display = np.clip(display, 600, 4000)  # clip to useful range
    display = cv2.normalize(display, None, 0, 255, cv2.NORM_MINMAX)
    display = np.uint8(display)

    # FIX: Apply colormap — much more readable than grayscale
    colormap = cv2.applyColorMap(display, cv2.COLORMAP_JET)

    # FIX: Show center distance with region averaging
    cx, cy = 320, 240
    dist = get_depth_at(depth_image, cx, cy)

    cv2.drawMarker(colormap, (cx, cy), (255, 255, 255),
                   cv2.MARKER_CROSS, 20, 2)

    if dist > 0:
        label = f"Center: {dist}mm  ({dist/10:.1f}cm)"
        color = (255, 255, 255)
    else:
        label = "Center: OUT OF RANGE"
        color = (0, 0, 255)

    cv2.putText(colormap, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Depth Colormap", colormap)

    if cv2.waitKey(1) == ord('q'):
        break

depth_stream.stop()
openni2.unload()
cv2.destroyAllWindows()