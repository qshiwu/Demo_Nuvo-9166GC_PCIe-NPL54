import cv2
print(cv2.getBuildInformation())

import cv2
import matplotlib.pyplot as plt

# Open video file
cap = cv2.VideoCapture("/dev/video102")  # For a specific camera

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()  # Read each frame
    if not ret:
        break

    cv2.imshow("Video", frame)  # Show frame

    if cv2.waitKey(25) & 0xFF == ord("q"):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
