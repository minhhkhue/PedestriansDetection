import numpy as np
import cv2
import matplotlib as plt

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cv2.startWindowThread()

vid = cv2.VideoCapture('datasets/test.mp4')

while(True):
    ret, frame = vid.read()
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    boxes, weights = hog.detectMultiScale(frame, winStride = (8, 8))
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    for(xA, yA, xB, yB) in boxes:
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0,  255, 0))
        cv2.putText(frame, 'Human', (xA - 2, yA - 2), 1, 0.75, (255, 255, 0), 1)

    cv2.imshow("Pedestrian Detection", frame)

        # Check for the 'q' key to quit the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # Release the capture and close all OpenCV windows
vid.release()
cv2.destroyAllWindows()
cv2.waitKey(1)