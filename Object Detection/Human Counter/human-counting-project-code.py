import cv2
import numpy as np

net = cv2.dnn.readNetFromCaffe(
    'mobilenet_ssd/MobileNetSSD_deploy.prototxt', 
    'mobilenet_ssd/MobileNetSSD_deploy.caffemodel'
)

cap = cv2.VideoCapture(0)

PERSON_CLASS_ID = 15

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    person_count = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            class_id = int(detections[0, 0, i, 1])

            if class_id == PERSON_CLASS_ID:
                person_count += 1

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    text = f"Human Count: {person_count}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Human Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
