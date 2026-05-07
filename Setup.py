from openni import openni2
import numpy as np
import cv2

# Initialize OpenNI
openni2.initialize(r"C:\OpenNI2\Bin")

# Open device
dev = openni2.Device.open_any()

# Create depth stream only
depth_stream = dev.create_depth_stream()
depth_stream.start()

print("Depth stream started...")

cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)

while True:
    # Read frame
    frame = depth_stream.read_frame()

    # Convert data
    frame_data = frame.get_buffer_as_uint16()
    img = np.frombuffer(frame_data, dtype=np.uint16)

    # Reshape
    img = img.reshape((480, 640))

    # Normalize for display
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = np.uint8(img)

    # Show image
    cv2.imshow("Depth", img)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

depth_stream.stop()
openni2.unload()
cv2.destroyAllWindows()