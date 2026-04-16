import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# 1. DEFINE TRANSFORM (same as training, but no augmentation)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 2. LOAD MODEL ARCHITECTURE
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 1)

# 3. LOAD TRAINED WEIGHTS
model.load_state_dict(torch.load("D:/ML_model/model/cataract_model.pth"))
model.eval()

# 4. CLASS NAMES
classes = ['cataract', 'normal']

# 5. TAKE FOLDER INPUT FROM USER
folder_path = "D:/ML_model/images_to_process"

# 6. LOOP THROUGH IMAGES
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)

    try:
        # Open image
        image = Image.open(file_path).convert("RGB")

        # Preprocess
        image = transform(image).unsqueeze(0)

        # Predict
        with torch.no_grad():
            output = model(image)
            prob = torch.sigmoid(output).item()

        # Convert to label
        prediction = classes[1] if prob > 0.5 else classes[0]

        print(f"{file_name} → {prediction} (confidence: {prob:.2f})")

    except Exception as e:
        print(f"Skipping {file_name} (error: {e})")
