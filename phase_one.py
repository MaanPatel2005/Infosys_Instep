import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Load the pre-trained Faster R-CNN model
model = models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
model.eval()

# Define the transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

# Open a video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    frame_cpy = frame.copy()
    if not ret:
        break

    # Convert the frame to RGB (from BGR) and apply transformations
    frame_rgb = cv2.cvtColor(frame_cpy, cv2.COLOR_BGR2RGB)
    frame_tensor = transform(frame_rgb).unsqueeze(0)

    # Make predictions
    with torch.no_grad():
        outputs = model(frame_tensor)

    # Check if a person is detected
    person_detected = False
    for box, label in zip(outputs[0]['boxes'], outputs[0]['labels']):
        if label.item() == 1:  # The label for 'person' is 1
            person_detected = True
            break

    if person_detected:
        print("Person detected in the image.")
    else:
        print("No person detected in the image.")

    # Optional: Visualize the results
    # Draw bounding boxes on the image
    if person_detected:
        for box, label, score in zip(outputs[0]['boxes'], outputs[0]['labels'], outputs[0]['scores']):
            if label.item() == 1 and score.item() > 0.5:  # Consider a threshold to filter out low-confidence detections
                xmin, ymin, xmax, ymax = box
                xmin = int(xmin.item())
                ymin = int(ymin.item())
                xmax = int(xmax.item())
                ymax = int(ymax.item())
                center_x = int((xmin+xmax)/2)
                center_y = int((ymin+ymax)/2)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.circle(frame,  (center_x, center_y), 5, (0, 255, 0), 2)
                cv2.putText(frame, f'Human {score.item():.2f}', (xmin, ymin-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Faster R-CNN Detection', frame)

    # Quit with the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
