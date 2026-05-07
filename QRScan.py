import cv2

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

detector = cv2.QRCodeDetector()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    data, bbox, _ = detector.detectAndDecode(frame)

    if bbox is not None:

        bbox = bbox.astype(int)

        for i in range(len(bbox[0])):
            pt1 = tuple(bbox[0][i])
            pt2 = tuple(bbox[0][(i + 1) % len(bbox[0])])

            cv2.line(frame, pt1, pt2, (0,255,0), 2)

        if data:
            cv2.putText(
                frame,
                data,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )

            print("QR:", data)

    cv2.imshow("QR Scanner", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()