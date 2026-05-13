
from openni import openni2
import numpy as np
import cv2

# Initialize OpenNI2
openni2.initialize(r"C:\OpenNI2\Bin")

dev = openni2.Device.open_any()

depth_stream = dev.create_depth_stream()
depth_stream.start()

print("Depth stream started... Press Q to quit")


def get_depth_at(depth_image, x, y, region=10):
  
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


cv2.namedWindow("Depth Colormap", cv2.WINDOW_NORMAL)

while True:

    frame = depth_stream.read_frame()
    frame_data = frame.get_buffer_as_uint16()

    depth_image = np.frombuffer(frame_data, dtype=np.uint16).copy()
    depth_image = depth_image.reshape((480, 640))

    
    display = depth_image.copy().astype(np.float32)
    display = np.clip(display, 600, 4000)  # clip to useful range
    display = cv2.normalize(display, None, 0, 255, cv2.NORM_MINMAX)
    display = np.uint8(display)

    
    colormap = cv2.applyColorMap(display, cv2.COLORMAP_JET)

    
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