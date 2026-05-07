from openni import openni2
import numpy as np
import cv2

# Initialize OpenNI2
openni2.initialize(r"C:\OpenNI2\Bin")

# Open device
dev = openni2.Device.open_any()

# Create depth stream
depth_stream = dev.create_depth_stream()

# Start stream
depth_stream.start()

print("Depth stream started... Press Q to quit")

# Create window
cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)

while True:

    # Read depth frame
    frame = depth_stream.read_frame()

    # Get depth data
    frame_data = frame.get_buffer_as_uint16()

    # Convert to NumPy array
    depth_image = np.frombuffer(
        frame_data,
        dtype=np.uint16
    )

    # Reshape image
    depth_image = depth_image.reshape((480, 640))

    # Normalize for display
    depth_display = cv2.normalize(
        depth_image,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    depth_display = np.uint8(depth_display)

    # Show image
    cv2.imshow("Depth", depth_display)

    # Exit on Q
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

# Cleanup
depth_stream.stop()

openni2.unload()

cv2.destroyAllWindows()