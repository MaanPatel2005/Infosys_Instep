# https://www.kaggle.com/code/sciencelabwork/simple-application-of-movenet      This is documentation of the model we are using
import torch
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
confidence = 0.7

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Load PoseNet model from TensorFlow Hub
pose_net = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = pose_net.signatures['serving_default']

def get_keypoints(image):
    input_image = tf.convert_to_tensor(image, dtype=tf.float32)
    input_image = tf.image.resize_with_pad(input_image, 192, 192)
    input_image = input_image[tf.newaxis, ...]
    input_image = tf.cast(input_image, dtype=tf.int32)

    # Run model inference
    outputs = movenet(input_image)
    keypoints_with_scores = outputs['output_0'].numpy()

    return keypoints_with_scores[0, 0, :, :]

def get_direction(keypoints):
    nose = keypoints[0]
    left_eye = keypoints[1]
    right_eye = keypoints[2]
    left_ear = keypoints[3]
    right_ear = keypoints[4]

    if nose[2] > confidence and left_eye[2] > confidence and right_eye[2] > confidence:
        eye_diff = left_eye[1] - right_eye[1]
        if abs(eye_diff) < 0.03:  # Adjust threshold as needed
            return "Towards"
        elif eye_diff > 0:
            return "Left"
        else:
            return "Right"
    elif nose[2] > confidence and left_ear[2] > confidence and right_ear[2] > confidence:
        ear_diff = left_ear[1] - right_ear[1]
        if ear_diff > 0:
            return "Left"
        else:
            return "Right"
    return "Unknown"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    df = results.pandas().xyxy[0]

    # only keep the person class
    persons = df[df['name'] == 'person']

    for _, row in persons.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        conf = row['confidence']

        # Crop the person area from the frame
        person_img = frame[y1:y2, x1:x2]

        # Process the cropped image with PoseNet
        keypoints = get_keypoints(person_img)
        direction = get_direction(keypoints)

        label = f'Person: {conf:.2f}, {direction}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Draw keypoints
        for kp in keypoints:
            if kp[2] > 0.5:  # Only draw keypoints with high confidence
                cv2.circle(frame, (int(kp[1] * (x2 - x1) + x1), int(kp[0] * (y2 - y1) + y1)), 5, (0, 0, 255), -1)

        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        midpoint_x = (left_shoulder[1] + right_shoulder[1])/2
        midpoint_y = (left_shoulder[0] + right_shoulder[0])/2
        cv2.circle(frame, (int(midpoint_x * (x2 - x1) + x1), int(midpoint_y * (y2 - y1) + y1)), 5, (0, 255, 0), -1)
    cv2.imshow('YOLOv5 + PoseNet - Person Detection & Direction', frame)

    # Quit with q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
