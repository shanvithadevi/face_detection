import cv2
import datetime

# Load Haar Cascade for face detection (comes with OpenCV)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Force DirectShow backend for Windows
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not capturing frames.")
        break

    # Convert to grayscale (Haar works best on grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # More sensitive detection parameters
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,   # smaller step → more sensitive
        minNeighbors=3,     # lower → detects more, but may add false positives
        minSize=(50, 50)    # ignore very tiny detections
    )

    # Draw bounding boxes and labels
    for (x, y, w, h) in faces:
        # Approximate confidence based on face area relative to frame
        confidence = min(100, (w * h) / (frame.shape[0] * frame.shape[1]) * 1000)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"Person ({confidence:.1f}%)", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show face count
    cv2.putText(frame, f'Faces: {len(faces)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # Show timestamp
    cv2.putText(frame, str(datetime.datetime.now()), (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.imshow("Live Detection (Person + Confidence)", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
