# detect_webcam.py

import torch
import cv2
import numpy as np

# Load your trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='runs/train/asl_model2/weights/best.pt')
model.conf = 0.25  # Confidence threshold (optional tuning)

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 = default webcam

print("📷 Press 'q' to quit the webcam")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run inference
    results = model(img_rgb)

    # Render predictions directly on the frame
    annotated_frame = np.squeeze(results.render())  # render() returns a list

    # Show frame
    cv2.imshow('ASL Sign Detection', annotated_frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
