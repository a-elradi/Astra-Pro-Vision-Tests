from openni import openni2
import numpy as np
import cv2

# ---------------- OPENNI DEPTH ----------------
openni2.initialize(r"C:\OpenNI2\Bin")

dev = openni2.Device.open_any()

depth_stream = dev.create_depth_stream()
depth_stream.start()

# ---------------- RGB CAMERA ----------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open RGB camera")
    exit()

print("RGB + Depth started")

while True:

    # ---------- RGB ----------
    ret, color_frame = cap.read()

    if not ret:
        print("Failed to get RGB frame")
        break

    # ---------- DEPTH ----------
    depth_frame = depth_stream.read_frame()

    depth_data = depth_frame.get_buffer_as_uint16()

    depth_array = np.frombuffer(depth_data, dtype=np.uint16)
    depth_array = depth_array.reshape((480, 640))

    # Normalize depth for display
    depth_display = cv2.normalize(
        depth_array,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    depth_display = np.uint8(depth_display)

    # ---------- SHOW ----------
    cv2.imshow("RGB", color_frame)
    cv2.imshow("Depth", depth_display)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

# Cleanup
cap.release()

depth_stream.stop()

openni2.unload()

cv2.destroyAllWindows()