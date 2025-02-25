import cv2
from ultralytics import YOLO

# Download YOLOv8n (nano version)
model = YOLO("yolov8n.pt")  # This will automatically download the model if not found locally

# Open video file
cap = cv2.VideoCapture("/dev/video102")  # For a specific camera

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()  # Read each frame
    if not ret:
        break

    # Run YOLOv8 on the frame
    results = model(frame, classes=[0])

    # Show results
    annotated_frame = results[0].plot()
    # cv2.imshow("YOLOv8 - /dev/video100", annotated_frame)
    cv2.imshow("YOLOv8 - /dev/video100", annotated_frame)


    # # Loop through the results and find the largest box
    # for result in results:
    #     boxes = result.boxes  # Get bounding boxes

    #     for box in boxes:
    #         x1, y1, x2, y2 = box.xyxy[0]  # Get bounding box coordinates
    #         area = (x2 - x1) * (y2 - y1)  # Calculate the area of the bounding box
            
    #         if area > max_area:
    #             max_area = area
    #             largest_box = box  # Store the largest box



    # # If a largest box was found, print its details
    # if largest_box is not None:
    #     x1, y1, x2, y2 = map(int, largest_box.xyxy[0])  # Convert to int for drawing
    #     conf = largest_box.conf[0]  # Confidence score
    #     class_id = int(largest_box.cls[0])  # Class ID
    #     print(f"Largest Box: Class {class_id}, Confidence: {conf:.2f}, BBox: ({x1}, {y1}, {x2}, {y2})")

    


    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    # Wait for Q or ESC
    if key in [27, ord('q'), ord('Q')]:
        break

cap.release()
cv2.destroyAllWindows()
