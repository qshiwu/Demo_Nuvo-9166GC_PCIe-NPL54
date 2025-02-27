import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
import cv2
from screeninfo import get_monitors
from utility import *
import time

# A full size canvas
# Get the primary monitor's resolution    
monitor = get_monitors()[0]  # Assumes the first monitor is the primary one
print(f"Desktop resolution: {monitor.width}x{monitor.height}")
canvas = np.zeros((monitor.height, monitor.width , 3), dtype=np.uint8)
canvas[:, :] = [255, 0, 0]
    



class StateMachine:
    def __init__(self):
        self.state = 'recording'  # Initial state
        self.transitions = {
            'recording': self.recording_state,
            'prepare_sam2': self.prepare_sam2_state,
            'inference': self.inference_state,
            'exit': self.exit_state
        }
        

    def recording_state(self):
        print("State: Recording")
        
        # Define the folder where you want to save the images
        folder_name = "captured_frames"

        # Create the folder if it doesn't exist
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        
        # Simulate the recording process
        # After recording, transition to 'prepare_sam2' state
        cap102 = cv2.VideoCapture('/dev/video102')
        # Check if the video source is opened
        if not cap102.isOpened():
            print("Error: Could not open video source.")
            exit()

        frame_count = 0
        max_frame = 90
        while frame_count < max_frame:
            ret, frame = cap102.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            texted_frame = putTextWithBackground(frame, f"{self.state} {frame_count +1 } / {max_frame}" , (10, 100))
            cv2.imshow("WIN", texted_frame);
            key = cv2.waitKey(6) & 0xFF    
            if key in [27, ord('q'), ord('Q')]:        
                cv2.destroyAllWindows()  # Close window
                exit()
                
        
            # Save frame as an image in JPEG format            
            filename = os.path.join(folder_name, f"frame_{frame_count:03d}.jpg")
            cv2.imwrite(filename, frame)
            frame_count += 1                                                        
        self.state = 'prepare_sam2'

    def prepare_sam2_state(self):
        print("State: Preparing SAM2")
        # Simulate preparing SAM2 process
        # After preparing SAM2, transition to 'inference' state
        self.state = 'inference'
        time.sleep(3)

    def inference_state(self):
        print("State: Inference")
        # Simulate the inference process
        # After inference, transition to 'exit' state
        self.state = 'recording'
        time.sleep(3)

    def exit_state(self):
        print("State: Exiting")
        # Exit the state machine
        print("State machine has finished.")         
        time.sleep(3)
        cv2.destroyAllWindows()  # Close window
        exit()

    def run(self):
        while True:
            print(f"Current state: {self.state}")
            global canvas
            canvas = putTextWithBackground(canvas, self.state, (10, 100))
            cv2.imshow("WIN", canvas);
            key = cv2.waitKey(6) & 0xFF    
            if key in [27, ord('q'), ord('Q')]:        
                 self.state = 'exit'
            
            # Change states if needed
            self.transitions[self.state]()  # Call the corresponding state handler


# Create an instance of the StateMachine class
sm = StateMachine()
# Run the state machine
sm.run()
##
printf("!")




device = torch.device("cuda")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

sam2_checkpoint = "../checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    
    
def putTextWithBackground(image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, 
                          font_scale=1, text_color=(255, 255, 255), 
                          bg_color=(0, 0, 0), thickness=2, padding=15):
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Define rectangle coordinates
    x, y = position
    rect_top_left = (x - padding, y - text_height - padding)
    rect_bottom_right = (x + text_width + padding, y + baseline + padding-10)
    
    # Draw filled rectangle
    cv2.rectangle(image, rect_top_left, rect_bottom_right, bg_color, -1)
    
    # Put text on top of rectangle
    cv2.putText(image, text, position, font, font_scale, text_color, thickness)

    return image  # Return modified image


# image_path = "/absolute/path/to/folder/image.jpg"  # Use absolute path if needed


    
cap102 = cv2.VideoCapture('/dev/video102')
# Check if the video source is opened
if not cap102.isOpened():
    print("Error: Could not open video source.")
    exit()

frame_count = 0
while frame_count < 200:
    ret, frame = cap102.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Save frame as an image in JPEG format
    
    filename = os.path.join(folder_name, f"frame_{frame_count:03d}.jpg")
    cv2.imwrite(filename, frame)

    frame_count += 1


# Path to the image inside a folder
image_path = "videos/bedroom/00100.jpg"  # Use relative path

# Read the image
img = cv2.imread(image_path)

# Check if image was loaded correctly
if img is None:
    print("Error: Image not found!")
else:
    # Display the image
    while True:
        img = putTextWithBackground(img, "Hello, OpenCV!", (10, 100))
        cv2.imshow("WIN", img);
        key = cv2.waitKey(6) & 0xFF    
        if key in [27, ord('q'), ord('Q')]:        
            break
    
cv2.destroyAllWindows()  # Close window

