import warnings
warnings.filterwarnings("ignore", message="Overwriting tiny_vit_5m_224 in registry")

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

# Download YOLOv8n (nano version)
model102 = YOLO("yolov8n.pt", verbose=False)  # This will automatically download the model103 if not found locally
model103 = YOLO("yolov8n.pt", verbose=False)  # This will automatically download the model103 if not found locally

# print(model102.model) 


### Begin of SAM2 Initialization ###

# from sam2.build_sam import build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor

# current_user = os.getenv("USER")
# sam2_checkpoint = f"/home/{current_user}/Desktop/sam2/checkpoints/sam2.1_hiera_tiny.pt"
# model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

# select the device for computation
# if torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
#     device = torch.device("cpu")
# print(f"using device: {device}")

# if device.type == "cuda":
#     # use bfloat16 for the entire notebook
#     torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
#     # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
#     if torch.cuda.get_device_properties(0).major >= 8:
#         torch.backends.cuda.matmul.allow_tf32 = True
#         torch.backends.cudnn.allow_tf32 = True

# sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
# predictor = SAM2ImagePredictor(sam2_model)

### End of SAM2 Initialization ###


### Begin of MobileSAM ###
from mobile_sam import sam_model_registry, SamPredictor

current_user = os.getenv("USER")
sam_checkpoint = f"/home/{current_user}/Desktop/MobileSAM/weights/mobile_sam.pt"

model_type = "vit_t"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
sam.eval()

predictor = SamPredictor(sam)

### End of MobileSAM ###

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


# Flag to control thread execution
running = True
    
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
      
    
def display_annoted_frame102():

   # print("--------------")
   # print(cv2.cuda.getCudaEnabledDeviceCount())
   # print("--------------")

    global running, frame102_queue, annotated_frame102
    while running:     
        if not frame102_queue.empty():            
            frame = frame102_queue.get() 
            
            # Run YOLOv8 on the frame102 (only person detection)
            results102 = model102(frame, conf=0.40, classes=[0], verbose=False)
            
            # for sam2 _ sam2 is too huge for real time processing
            # switching to mobile sam
            # ratio = 0.3
            # resized_frame = cv2.cuda.resize(frame, (0, 0), fx=ratio, fy=ratio)
            # predictor.set_image(resized_frame)
            predictor.set_image(frame)
            blended = frame
                                         
            for result in results102:
                for box_yolo in result.boxes:                    
                                        
                    # print(box)
                    x_min, y_min, x_max, y_max = box_yolo.xyxy[0]  # Bounding box in (x_min, y_min, x_max, y_max) format
                    # print(f"BBox: [{x_min:.0f}, {y_min:.0f}, {x_max:.0f}, {y_max:.0f}]")
                    
                    input_box = np.array([x_min.cpu().item(), y_min.cpu().item(), x_max.cpu().item(), y_max.cpu().item()])
                    
                    masks, _, _ = predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_box[None, :],
                        multimask_output=False,
                    )


                    mask = masks[0]
                    # Convert mask to uint8 (0 or 255) if it's a binary mask
                    mask = (mask > 0.5).astype(np.uint8) * 255  # Threshold if necessary

                    # Ensure the mask and frame have the same size
                    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

                    # Convert the mask to 3 channels for blending (if it's single channel)
                    mask_colored = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)

                    alpha = 1
                    beta = 0.5
                    blended = cv2.addWeighted(blended, alpha , mask_colored, beta , 0)


                    #color = np.array([30/255, 144/255, 255/255, 0.6])
                    #h, w = masks[0].shape[-2:]
                    #mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

                
                
            # Show results
            annotated_frame102 = blended
            # annotated_frame102 = results102
            
        time.sleep(0.010)
    
      
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
 


   
# Create and Start Threads
# thread_display_canvas = threading.Thread(target=display_canvas)
# thread_display_canvas.start()

thread_display_frame102 = threading.Thread(target=display_frame102)
thread_display_frame102.start()

thread_display_annoted_frame102 = threading.Thread(target=display_annoted_frame102)
thread_display_annoted_frame102.start()


thread_display_frame103 = threading.Thread(target=display_frame103)
thread_display_frame103.start()

thread_display_annoted_frame103 = threading.Thread(target=display_annoted_frame103)
thread_display_annoted_frame103.start()


# Display the resulting frame    
#  frame102   | frame102 w/ AI
# —————————————————————————————
#  frame103   | frame103 w/ AI    
# Display every video stream on canvas and listen to keyboard event

while True:
    
    # frame 102
    canvas[0:monitor.height//2, 0:monitor.width//2] = cv2.resize(frame102, (monitor.width//2, monitor.height//2))
    canvas[0:monitor.height//2, monitor.width//2:monitor.width] = cv2.resize(annotated_frame102, (monitor.width//2, monitor.height//2))
    
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
thread_display_annoted_frame102.join()
thread_display_annoted_frame103.join()

# When everything done, release the cap102 and cap103
cap102.release()
cap103.release()
cv2.destroyAllWindows()
