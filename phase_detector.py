import torch
from torchvision import models, transforms
from PIL import Image
import cv2
import os
import torch.nn as nn
import numpy as np

# Define class names
class_names = ['agent phase', 'buy phase', 'game play']

# Load the model
model = models.resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("D:/dual cast/training/phase_classifier.pt", map_location=torch.device('cpu')))
model.eval()

# Define image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_image(img):
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, 1).item()
        confidence = torch.softmax(output, dim=1)[0][predicted_class].item()
    return class_names[predicted_class], confidence

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        prediction, confidence = predict_image(frame)

        # Overlay prediction on frame
        text = f"{prediction} ({confidence:.2f})"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Show frame
        cv2.imshow('Phase Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, 1).item()
        confidence = torch.softmax(output, dim=1)[0][predicted_class].item()

    print(f"Prediction: {class_names[predicted_class]} (Confidence: {confidence:.2f})")

# ====== Choose between image or video ======
input_path = "D:/dual cast/sample_video.mp4"  # Change this to video path if needed

if input_path.lower().endswith(('.mp4', '.avi', '.mov')):
    process_video(input_path)
else:
    process_image(input_path)
