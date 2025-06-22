from model import SignLanguageCNN
from dataloader import get_dataloaders
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SignLanguageCNN()
model.load_state_dict(torch.load("../models/model_epoch_10.pth"))
model.to(device)
model.eval()

_, test_loader, classes = get_dataloaders("../data")

correct = 0
total = 0

with torch.no_grad():
  for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)
    outputs = model.forward(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
  
print(f"Accuracy: {100 * correct / total:.2f}%")