import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import sys
import torch
import numpy as np
import cv2
import threading
import time
import queue
import torch

from utility import *
from screeninfo import get_monitors
from ultralytics import YOLO

### Initialize YOLOv8n ###
WIN_NAME = "/dev/video103 __ Live | YOLOv8n" 
padding_h = 20
# Download YOLOv8n (nano version)
model103 = YOLO("yolov8n.pt", verbose=False)  # This will automatically download the model103 if not found locally




# Queue to share frames between threads (size=1 ensures old frames are dropped)
frame103_queue = queue.Queue(maxsize=1)

cap103 = cv2.VideoCapture('/dev/video103')

cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)

if not cap103.isOpened():
    print("Neither /dev/video102 nor /dev/video103 is not streaming")
    exit()
else:
    # Get 1st Frame
    ret103, frame103 = cap103.read()
    annotated_frame103 = frame103
    
    # Get the primary monitor's resolution    
    monitor = get_monitors()[0]  # Assumes the first monitor is the primary one
    print(f"Desktop resolution: {monitor.width}x{monitor.height}")
    cv2.resizeWindow(WIN_NAME,monitor.width,monitor.height//2-padding_h)   
    cv2.moveWindow(WIN_NAME,0,monitor.height//2)
    # A full size canvas
    canvas = np.zeros((monitor.height//2, monitor.width , 3), dtype=np.uint8)
    # canvas = np.zeros((monitor.height, monitor.width , 3), dtype=np.uint8)
    canvas[:, :] = [255, 0, 0]    


# Flag to control thread execution
running = True
    
def display_frame103():
    global frame103, running, frame103_queue
    while running and cap103.isOpened():        
        ret103, frame103 = cap103.read()
        # if frame is read correctly ret is True
        if not ret103:            
            print("Can't receive frame103 . Exiting ...")
            break
        
        # Add frame to queue (drop old frames if needed)
        if not frame103_queue.full():
            frame103_queue.put(frame103)


def display_annoted_frame103():
    global running, frame103_queue, annotated_frame103
    while running:     
        if not frame103_queue.empty():            
            frame = frame103_queue.get()            
            # Run YOLOv8 on the frame103 (only person detection)
            results103 = model103(frame, conf=0.45, classes=[0], verbose=False)
            # Show results
            annotated_frame103 = results103[0].plot()
        time.sleep(0.010)
 


   
thread_display_frame103 = threading.Thread(target=display_frame103)
thread_display_frame103.start()

thread_display_annoted_frame103 = threading.Thread(target=display_annoted_frame103)
thread_display_annoted_frame103.start()


# Display the resulting frame    
# —————————————————————————————
#  frame103   | frame103 w/ AI    
# Display every video stream on canvas and listen to keyboard event

while True:
    
    # frame 103
    canvas[0:monitor.height//2, 0:monitor.width//2] = cv2.resize(frame103, (monitor.width//2, monitor.height//2))
    canvas[0:monitor.height//2, monitor.width//2:monitor.width] = cv2.resize(annotated_frame103, (monitor.width//2, monitor.height//2))
    
    # cv2.line(canvas, (0, monitor.height//2), (monitor.width, monitor.height//2), (200, 200, 200), 1)
    cv2.line(canvas, (monitor.width//2, 0), (monitor.width//2, monitor.height), (200, 200, 200), 1)
    cv2.imshow(WIN_NAME, canvas)    
    
    # Wait for key press: Q or ESC
    key = cv2.waitKey(6) & 0xFF    
    if key in [27, ord('q'), ord('Q')]:
        running = False
        break
    


# Wait for Threads to Finish
thread_display_frame103.join()
thread_display_annoted_frame103.join()

# When everything done, release the cap102 and cap103
cap103.release()
cv2.destroyAllWindows()
