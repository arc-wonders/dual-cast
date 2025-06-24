import torch
from torchvision import models, transforms
from PIL import Image
import os
import torch.nn as nn

# Define class names in the **same order** as in your training folders
class_names = ['agent phase', 'buy phase', 'game play']

# Load the model
model = models.resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load("D:/dual cast/training/phase_classifier.pt", map_location=torch.device('cpu')))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load and preprocess the image
image_path = "D:/dual cast/random.jpeg"  # change to your test image path
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0)

# Inference
with torch.no_grad():
    output = model(input_tensor)
    predicted_class = torch.argmax(output, 1).item()
    confidence = torch.softmax(output, dim=1)[0][predicted_class].item()

# Print result
print(f"Prediction: {class_names[predicted_class]} (Confidence: {confidence:.2f})")
