import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    average_precision_score
)

from dataset import train_loader, val_loader, test_loader
from model import BrainTumorNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

model = BrainTumorNN().to(device)

# He Initialization
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)

model.apply(initialize_weights)

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0005,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=5
)

epochs = 100

train_losses = []
train_accuracies = []

best_accuracy = 0

for epoch in range(epochs):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        predictions = (torch.sigmoid(outputs) > 0.5).float()

        correct += (predictions == labels).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / len(train_loader)

    epoch_accuracy = 100 * correct / total

    train_losses.append(epoch_loss)

    train_accuracies.append(epoch_accuracy)

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {epoch_loss:.4f} "
        f"Train Accuracy: {epoch_accuracy:.2f}%"
    )

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)

            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)

            predictions = (torch.sigmoid(outputs) > 0.5).float()

            correct += (predictions == labels).sum().item()

            total += labels.size(0)

    val_accuracy = 100 * correct / total

    scheduler.step(val_accuracy)

    print(f"Validation Accuracy: {val_accuracy:.2f}%")

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "brain_tumor_model.pth"
        )

        print("Best model saved!")

model.load_state_dict(
    torch.load("brain_tumor_model.pth", map_location=device)
)

model.eval()

correct = 0
total = 0

y_true = []
y_pred = []
y_scores = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        outputs = model(images)

        probabilities = torch.sigmoid(outputs)

        predictions = (probabilities > 0.5).float()

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        y_true.extend(labels.cpu().numpy().flatten())
        y_pred.extend(predictions.cpu().numpy().flatten())
        y_scores.extend(probabilities.cpu().numpy().flatten())

test_accuracy = 100 * correct / total

print("\n==============================")
print(f"Test Accuracy : {test_accuracy:.2f}%")
print("==============================")

print("\nClassification Report:\n")

print(classification_report(y_true, y_pred))

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Tumor", "Tumor"]
)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png", dpi=300)

plt.show()

precision, recall, _ = precision_recall_curve(
    y_true,
    y_scores
)

ap = average_precision_score(
    y_true,
    y_scores
)

plt.figure(figsize=(6,6))

plt.plot(
    recall,
    precision,
    linewidth=2,
    label=f"AP = {ap:.3f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(True)

plt.savefig("precision_recall_curve.png", dpi=300)

plt.show()

plt.figure(figsize=(7,5))

plt.plot(
    train_losses,
    linewidth=2,
    color="blue"
)

plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

plt.savefig("training_loss.png", dpi=300)

plt.show()

plt.figure(figsize=(7,5))

plt.plot(
    train_accuracies,
    linewidth=2,
    color="green"
)
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.savefig("training_accuracy.png", dpi=300)
plt.show()
print("\n==============================")
print("Training Complete!")
print(f"Best Validation Accuracy : {best_accuracy:.2f}%")
print(f"Final Test Accuracy      : {test_accuracy:.2f}%")
print("==============================")