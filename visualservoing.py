from openni import openni2
import numpy as np
import cv2

# ---------------- OPENNI DEPTH ----------------
openni2.initialize(r"C:\OpenNI2\Bin")

dev = openni2.Device.open_any()

depth_stream = dev.create_depth_stream()
depth_stream.start()

# ---------------- RGB CAMERA ----------------
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ---------------- DISTANCE FUNCTION ----------------
def get_distance_at_pixel(depth_image, x, y):

    if x < 0 or y < 0:
        return 0

    if y >= depth_image.shape[0] or x >= depth_image.shape[1]:
        return 0

    return depth_image[y, x]

# ---------------- VISUAL SERVO ----------------
def visual_servo(box_cx, frame_width, Kp=0.005):

    frame_center = frame_width / 2

    # Error from center
    error = box_cx - frame_center

    # Proportional control
    correction = error * Kp

    # Dead zone
    threshold = 20

    aligned = abs(error) < threshold

    if aligned:
        correction = 0

    return correction, aligned

# ---------------- BOX DETECTION ----------------
def detect_box(rgb_frame, depth_image):

    gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)

    edges = cv2.Canny(blur, 50, 150)

    kernel = np.ones((5,5), np.uint8)

    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    best = None
    best_area = 0

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area < 15000:
            continue

        peri = cv2.arcLength(cnt, True)

        approx = cv2.approxPolyDP(
            cnt,
            0.02 * peri,
            True
        )

        # Rectangle only
        if len(approx) == 4:

            x, y, w, h = cv2.boundingRect(approx)

            rect_area = w * h

            if rect_area > best_area:

                best_area = rect_area
                best = (x, y, w, h)

    if best:

        x, y, w, h = best

        cx = x + w // 2
        cy = y + h // 2

        # Convert RGB coords -> Depth coords
        dx = int(cx * 640 / rgb_frame.shape[1])
        dy = int(cy * 480 / rgb_frame.shape[0])

        distance = get_distance_at_pixel(depth_image, dx, dy)

        return x, y, w, h, cx, cy, distance

    return None

# ---------------- MAIN LOOP ----------------
while True:

    # RGB FRAME
    ret, frame = cap.read()

    if not ret:
        break

    # DEPTH FRAME
    depth_frame = depth_stream.read_frame()

    depth_data = depth_frame.get_buffer_as_uint16()

    depth_image = np.frombuffer(
        depth_data,
        dtype=np.uint16
    )

    depth_image = depth_image.reshape((480, 640))

    # DETECT BOX
    result = detect_box(frame, depth_image)

    if result:

        x, y, w, h, cx, cy, distance = result

        # Draw box
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0,255,0),
            3
        )

        # Draw center
        cv2.circle(
            frame,
            (cx, cy),
            5,
            (0,0,255),
            -1
        )

        # Distance text
        cv2.putText(
            frame,
            f"Distance: {distance} mm",
            (x, y - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

        # ---------------- VISUAL SERVO ----------------
        correction, aligned = visual_servo(
            cx,
            frame.shape[1]
        )

        # Servo status
        if aligned:

            status = "CENTERED"

        elif correction > 0:

            status = "TURN RIGHT"

        else:

            status = "TURN LEFT"

        # Display status
        cv2.putText(
            frame,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,0,0),
            2
        )

        # Display correction
        cv2.putText(
            frame,
            f"Correction: {correction:.2f}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,0,0),
            2
        )

        # Console output
        print(f"{status} | Correction: {correction:.2f}")

    # Show frame
    cv2.imshow("Visual Servoing", frame)

    # Quit
    if cv2.waitKey(1) == ord('q'):
        break

# ---------------- CLEANUP ----------------
cap.release()

depth_stream.stop()

openni2.unload()

cv2.destroyAllWindows()