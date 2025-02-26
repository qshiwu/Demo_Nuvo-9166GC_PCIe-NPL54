# pip install screeninfo
from screeninfo import get_monitors

import numpy as np
import cv2

cap102 = cv2.VideoCapture('/dev/video102')
cap103 = cv2.VideoCapture('/dev/video103')

cv2.namedWindow("WIN", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("WIN", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)




if not cap102.isOpened() or not cap103.isOpened():
    print("Neither /dev/video102 nor /dev/video103 is not streaming")
    exit()
else:
    # Get the primary monitor's resolution    
    monitor = get_monitors()[0]  # Assumes the first monitor is the primary one
    print(f"Desktop resolution: {monitor.width}x{monitor.height}")
    
    # A full size canvas
    canvas = np.zeros((monitor.height, monitor.width , 3), dtype=np.uint8)
    canvas[:, :] = [255, 0, 0]
        

    
while True:

    
    # cap103ture frame-by-frame
    ret103, frame103 = cap103.read()
    ret102, frame102 = cap102.read()

    # if frame is read correctly ret is True
    if not ret103:
        print("Can't receive frame103 (stream end?). Exiting ...")
        break
    
    
    # Our operations on the frame come here
    gray103 = cv2.cvtColor(frame103, cv2.COLOR_BGR2GRAY)
    resizedGray103 = cv2.resize(gray103, (monitor.width//2, monitor.height//2))
    # Display the resulting frame
    # cv2.imshow("WIN", resizedGray103)
    # canvas[y:y+height, x:x+width]
    canvas[0:monitor.height//2, 0:monitor.width//2] = cv2.resize(frame102, (monitor.width//2, monitor.height//2))
    canvas[monitor.height//2:monitor.height, 0:monitor.width//2] = cv2.resize(frame103, (monitor.width//2, monitor.height//2))
    # canvas[0:1080, 0:1920] = cv2.resize(frame103,(20,20))
    cv2.line(canvas, (0, monitor.height//2), (monitor.width, monitor.height//2), (200, 200, 200), 2)
    cv2.line(canvas, (monitor.width//2, 0), (monitor.width//2, monitor.height), (200, 200, 200), 2)

    cv2.imshow("WIN", canvas)
    

    
    # Wait for key press: Q or ESC
    key = cv2.waitKey(1) & 0xFF    
    if key in [27, ord('q'), ord('Q')]:
        break





# When everything done, release the cap102 and cap103
cap102.release()
cap103.release()
cv2.destroyAllWindows()