# test_yolo.py — run this first
from ultralytics import YOLO
import cv2

# Load pretrained model — downloads automatically (~6MB)
model = YOLO("yolov8n.pt")  # n = nano, fastest on Jetson

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame, verbose=False)

    # Draw results
    annotated = results[0].plot()

    cv2.imshow("YOLOv8 Detection", annotated)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()