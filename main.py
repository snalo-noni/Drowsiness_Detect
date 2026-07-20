import cv2
import numpy as np
import time
import winsound  # Native audio alert for Windows
import os

# Get current script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

face_xml = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
eye_xml = os.path.join(BASE_DIR, "haarcascade_eye.xml")

# Load Cascades
face_cascade = cv2.CascadeClassifier(face_xml)
eye_cascade = cv2.CascadeClassifier(eye_xml)

# Verify files loaded properly
if face_cascade.empty() or eye_cascade.empty():
    print(
        "[ERROR] Could not load XML files. Make sure 'haarcascade_frontalface_default.xml' and 'haarcascade_eye.xml' are in your project folder!")
    exit()

# Thresholds & Counters
CLOSED_EYE_FRAMES = 20  # Consecutive frames eyes are closed before alert
closed_counter = 0

cap = cv2.VideoCapture(0)
prev_time = time.time()

print("[INFO] Starting Drowsiness Detection System. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Camera feed unavailable.")
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    status_text = "Status: Alert"
    status_color = (0, 255, 0)  # Green

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect eyes inside face region
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 2)

        if len(eyes) < 2:
            closed_counter += 1
            if closed_counter >= CLOSED_EYE_FRAMES:
                status_text = "DROWSINESS DETECTED!"
                status_color = (0, 0, 255)  # Red
                try:
                    winsound.Beep(1000, 200)
                except Exception:
                    pass
        else:
            closed_counter = 0

    # UI Overlay
    cv2.rectangle(frame, (10, 10), (320, 100), (0, 0, 0), -1)
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, status_text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    cv2.imshow("Driver Drowsiness Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()