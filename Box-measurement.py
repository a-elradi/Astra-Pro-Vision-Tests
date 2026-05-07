import cv2
import numpy as np
from openni import openni2

# ---------------- DEPTH ----------------
openni2.initialize(r"C:\OpenNI2\Bin")

dev = openni2.Device.open_any()

depth_stream = dev.create_depth_stream()
depth_stream.start()

# ---------------- RGB ----------------
cap = cv2.VideoCapture(1)

# ---------------- MAIN ----------------
while True:

    ret, frame = cap.read()

    if not ret:
        break

    # DEPTH
    depth_frame = depth_stream.read_frame()

    depth_data = depth_frame.get_buffer_as_uint16()

    depth_image = np.frombuffer(depth_data, dtype=np.uint16)
    depth_image = depth_image.reshape((480, 640))

    # PROCESS IMAGE
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5,5), 0)

    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area < 5000:
            continue

        # Approx shape
        peri = cv2.arcLength(cnt, True)

        approx = cv2.approxPolyDP(
            cnt,
            0.02 * peri,
            True
        )

        # RECTANGLE ONLY
        if len(approx) == 4:

            x, y, w, h = cv2.boundingRect(approx)

            cx = x + w // 2
            cy = y + h // 2

            # Convert RGB coords -> Depth coords
            dx = int(cx * 640 / frame.shape[1])
            dy = int(cy * 480 / frame.shape[0])

            distance_mm = depth_image[dy, dx]

            # Draw
            cv2.drawContours(frame, [approx], -1, (0,255,0), 3)

            cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)

            cv2.putText(
                frame,
                f"Distance: {distance_mm} mm",
                (x, y - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,0),
                2
            )

            cv2.putText(
                frame,
                f"W:{w}px H:{h}px",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,0,0),
                2
            )

    cv2.imshow("Box Measurement", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()

depth_stream.stop()

openni2.unload()

cv2.destroyAllWindows()