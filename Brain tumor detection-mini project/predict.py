import os
import random
import torch
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

from model import BrainTumorNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BrainTumorNN().to(device)

model.load_state_dict(
    torch.load("brain_tumor_model.pth", map_location=device)
)

model.eval()

transform = transforms.Compose([

    transforms.Resize((128, 128)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )

])
folder = random.choice(["Dataset/yes", "Dataset/no"])

filename = random.choice(os.listdir(folder))

image_path = os.path.join(folder, filename)

image = Image.open(image_path).convert("RGB")

actual = "Tumor" if folder.endswith("yes") else "No Tumor"

input_image = transform(image)

input_image = input_image.unsqueeze(0).to(device)

with torch.no_grad():

    output = model(input_image)

    probability = torch.sigmoid(output).item()

prediction = "Tumor" if probability >= 0.5 else "No Tumor"

confidence = probability if prediction == "Tumor" else (1 - probability)

plt.figure(figsize=(6,6))

plt.imshow(image)

plt.axis("off")

plt.title(
    f"Actual : {actual}\n"
    f"Prediction : {prediction}\n"
    f"Confidence : {confidence*100:.2f}%"
)

plt.show()

print("=" * 40)
print("Actual      :", actual)
print("Prediction  :", prediction)
print("Confidence  : {:.2f}%".format(confidence * 100))
print("=" * 40)