from openni import openni2
from openni import _openni2 as c_api
import numpy as np
import cv2

# Initialize OpenNI
openni2.initialize(r"C:\OpenNI2\Bin")

# Open device
dev = openni2.Device.open_any()

# ---------------- DEPTH STREAM ----------------
depth_stream = dev.create_depth_stream()

depth_stream.set_video_mode(
    c_api.OniVideoMode(
        pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM,
        resolutionX=640,
        resolutionY=480,
        fps=30
    )
)

# ---------------- RGB STREAM ----------------
color_stream = dev.create_color_stream()

color_stream.set_video_mode(
    c_api.OniVideoMode(
        pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888,
        resolutionX=640,
        resolutionY=480,
        fps=30
    )
)

# Start streams
depth_stream.start()
color_stream.start()

print("RGB + Depth started... Press Q to quit")

# Create windows
cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)
cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)

while True:

    # ---------- DEPTH ----------
    depth_frame = depth_stream.read_frame()
    depth_data = depth_frame.get_buffer_as_uint16()

    depth_array = np.frombuffer(depth_data, dtype=np.uint16)
    depth_array = depth_array.reshape((480, 640))

    # Normalize for display
    depth_display = cv2.normalize(
        depth_array,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    depth_display = np.uint8(depth_display)

    # ---------- RGB ----------
    color_frame = color_stream.read_frame()
    color_data = color_frame.get_buffer_as_uint8()

    color_array = np.frombuffer(color_data, dtype=np.uint8)
    color_array = color_array.reshape((480, 640, 3))

    # Convert RGB -> BGR for OpenCV
    color_array = cv2.cvtColor(color_array, cv2.COLOR_RGB2BGR)

    # ---------- SHOW ----------
    cv2.imshow("Depth", depth_display)
    cv2.imshow("RGB", color_array)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

# Cleanup
depth_stream.stop()
color_stream.stop()

openni2.unload()
cv2.destroyAllWindows()