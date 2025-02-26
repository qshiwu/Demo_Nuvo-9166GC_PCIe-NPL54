import numpy as np
import cv2 as cv

cap102 = cv.VideoCapture('/dev/video102')
cap103 = cv.VideoCapture('/dev/video103')

if not cap103.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # cap103ture frame-by-frame
    ret, frame = cap103.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv.imshow('frame', gray)


    # Wait for key press: Q or ESC
    key = cv.waitKey(1) & 0xFF    
    if key in [27, ord('q'), ord('Q')]:
        break





# When everything done, release the cap102 and cap103
cap102.release()
cap103.release()
cv.destroyAllWindows()