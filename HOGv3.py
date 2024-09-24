import numpy as np
import cv2

# Initialize HOG descriptor
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cv2.startWindowThread()

# Path to the input video file
input_video_path = "datasets/test.mp4"

# Open the input video
cap = cv2.VideoCapture(input_video_path)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Resize frame for faster processing (optional)
    frame = cv2.resize(frame, (840, 580))

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect people using HOG
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))

    # Convert boxes to numpy array for easier manipulation
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    # Draw bounding boxes around detected people
    for xA, yA, xB, yB in boxes:
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Pedestrian Detection", frame)

    # Check for the 'q' key to quit the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)  # This line is necessary to properly close the OpenCV windows