import numpy as np
import cv2
import threading
import time
import queue
# pip install screeninfo
from screeninfo import get_monitors
from ultralytics import YOLO


# Download YOLOv8n (nano version)
model = YOLO("yolov8n.pt")  # This will automatically download the model if not found locally

# Queue to share frames between threads (size=1 ensures old frames are dropped)
frame102_queue = queue.Queue(maxsize=1)
frame103_queue = queue.Queue(maxsize=1)

cap102 = cv2.VideoCapture('/dev/video102')
cap103 = cv2.VideoCapture('/dev/video103')

cv2.namedWindow("WIN", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("WIN", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


if not cap102.isOpened() or not cap103.isOpened():
    print("Neither /dev/video102 nor /dev/video103 is not streaming")
    exit()
else:
    # Get 1st Frame
    ret103, frame103 = cap103.read()
    annotated_frame103 = frame103
    ret102, frame102 = cap102.read()
    annotated_frame102 = frame102
    
    # Get the primary monitor's resolution    
    monitor = get_monitors()[0]  # Assumes the first monitor is the primary one
    print(f"Desktop resolution: {monitor.width}x{monitor.height}")
    
    # A full size canvas
    canvas = np.zeros((monitor.height, monitor.width , 3), dtype=np.uint8)
    canvas[:, :] = [255, 0, 0]
        

    
# while True:

#     # cap103ture frame-by-frame
#     ret103, frame103 = cap103.read()
#     ret102, frame102 = cap102.read()

#     # if frame is read correctly ret is True
#     if not ret102:
#         print("Can't receive frame102 . Exiting ...")
#         break

#     if not ret103:
#         print("Can't receive frame103 . Exiting ...")
#         break
    
#     # Run YOLOv8 on the frame103 (only person detection)
#     results103 = model(frame103, classes=[0])
#      # Show results
#     annotated_frame103 = results103[0].plot()

    
#     # Our operations on the frame come here
#     gray103 = cv2.cvtColor(frame103, cv2.COLOR_BGR2GRAY)
#     resizedGray103 = cv2.resize(gray103, (monitor.width//2, monitor.height//2))
    
#     # Display the resulting frame    
#     #  frame102   | frame102 w/ AI
#     # —————————————————————————————
#     #  frame103   | frame103 w/ AI
    
#     # frame 102
#     canvas[0:monitor.height//2, 0:monitor.width//2] = cv2.resize(frame102, (monitor.width//2, monitor.height//2))
    
#     # frame 103
#     canvas[monitor.height//2:monitor.height, 0:monitor.width//2] = cv2.resize(frame103, (monitor.width//2, monitor.height//2))
#     canvas[monitor.height//2:monitor.height, monitor.width//2:monitor.width] = cv2.resize(annotated_frame103, (monitor.width//2, monitor.height//2))
    
    
#     # canvas[0:1080, 0:1920] = annotated_frame_103
    
#     cv2.line(canvas, (0, monitor.height//2), (monitor.width, monitor.height//2), (200, 200, 200), 1)
#     cv2.line(canvas, (monitor.width//2, 0), (monitor.width//2, monitor.height), (200, 200, 200), 1)
#     cv2.imshow("WIN", canvas)
    

    
#     # Wait for key press: Q or ESC
#     key = cv2.waitKey(1) & 0xFF    
#     if key in [27, ord('q'), ord('Q')]:
#         break





# Flag to control thread execution
running = True

def display_canvas():    
    global running         
    while running:
        # cv2.imshow("WIN", canvas)
        print("x")
        # Wait for key press: Q or ESC
        # key = cv2.waitKey(1) & 0xFF    
        # if key in [27, ord('q'), ord('Q')]:
        #     running = False
        #     break
        # if keyboard.is_pressed('q'):
        #     print("Key 'q' pressed, exiting thread...")
        #     break
        time.sleep(0.1)  # Avoid high CPU usage
    
    
def display_frame102():
    global frame102, running, frame102_queue
    while running and cap102.isOpened():        
        ret102, frame102 = cap102.read()
        # if frame is read correctly ret is True
        if not ret102:            
            print("Can't receive frame102 . Exiting ...")
            break
        
        # Add frame to queue (drop old frames if needed)
        if not frame102_queue.full():
            frame102_queue.put(frame102)
      
      
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
            results103 = model(frame103, classes=[0])
            # Show results
            annotated_frame103 = results103[0].plot()
        time.sleep(0.006)
    
   
# Create and Start Threads
# thread_display_canvas = threading.Thread(target=display_canvas)
# thread_display_canvas.start()

thread_display_frame102 = threading.Thread(target=display_frame102)
thread_display_frame102.start()

thread_display_frame103 = threading.Thread(target=display_frame103)
thread_display_frame103.start()

thread_display_annoted_frame103 = threading.Thread(target=display_annoted_frame103)
thread_display_annoted_frame103.start()


# Display the Canvas
while True:
    # Display the resulting frame    
    #  frame102   | frame102 w/ AI
    # —————————————————————————————
    #  frame103   | frame103 w/ AI
    
    # frame 102
    canvas[0:monitor.height//2, 0:monitor.width//2] = cv2.resize(frame102, (monitor.width//2, monitor.height//2))
    
    # frame 103
    canvas[monitor.height//2:monitor.height, 0:monitor.width//2] = cv2.resize(frame103, (monitor.width//2, monitor.height//2))
    canvas[monitor.height//2:monitor.height, monitor.width//2:monitor.width] = cv2.resize(annotated_frame103, (monitor.width//2, monitor.height//2))

    cv2.line(canvas, (0, monitor.height//2), (monitor.width, monitor.height//2), (200, 200, 200), 1)
    cv2.line(canvas, (monitor.width//2, 0), (monitor.width//2, monitor.height), (200, 200, 200), 1)
    cv2.imshow("WIN", canvas)    
    
    # Wait for key press: Q or ESC
    key = cv2.waitKey(6) & 0xFF    
    if key in [27, ord('q'), ord('Q')]:
        running = False
        break
    


# Wait for Threads to Finish
# thread_display_canvas.join()
thread_display_frame102.join()
thread_display_frame103.join()
thread_display_annoted_frame103.join()


# When everything done, release the cap102 and cap103
cap102.release()
cap103.release()
cv2.destroyAllWindows()