import torch
from torchvision import models, transforms
from PIL import Image
import os
import torch.nn as nn

# Define class names (same order as training)
class_names = ['agent phase', 'buy phase', 'game play']

# Load model
model = models.resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load("D:/dual cast/training/phase_classifier.pt", map_location=torch.device('cpu')))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Folder containing test images
image_folder = "D:/dual cast/test"  # ✅ Change this folder path as needed

# Supported image extensions
valid_exts = ['.jpg', '.jpeg', '.png']

# Predict for all images
for filename in os.listdir(image_folder):
    if any(filename.lower().endswith(ext) for ext in valid_exts):
        image_path = os.path.join(image_folder, filename)
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = torch.argmax(output, 1).item()
            confidence = torch.softmax(output, dim=1)[0][predicted_class].item()

        print(f"[✅] {filename} --> {class_names[predicted_class]} (Confidence: {confidence:.2f})")
    else:
        print(f"[❌] {filename} --> skipped (unsupported format)")
