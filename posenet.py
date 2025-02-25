import cv2
import tensorflow as tf
import numpy as np

# Load the PoseNet model from TensorFlow Hub
posenet_model = tf.saved_model.load('https://tfhub.dev/tensorflow/posenet/1/default/1')


cap = cv2.VideoCapture('/dev/video102')

while True:
    # Capture frame from webcam
    ret, frame = cap.read()

    # Convert frame to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Normalize image to [-1, 1]
    input_image = tf.convert_to_tensor(image, dtype=tf.float32)
    input_image = tf.image.resize(input_image, (257, 257))
    input_image = input_image / 127.5 - 1

    # Run the model and get predictions
    input_image = input_image[tf.newaxis, ...]
    keypoints = posenet_model(input_image)

    # Process keypoints and draw them on the frame
    for keypoint in keypoints:
        x, y, score = keypoint[:3]
        if score > 0.5:  # Minimum confidence
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    # Display the frame with keypoints
    cv2.imshow('PoseNet', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

