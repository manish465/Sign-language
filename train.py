import torch
import torch.nn as nn
import torch.optim as optim

from model import SignLanguageCNN
from dataloader import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SignLanguageCNN().to(device=device)

train_loader, val_loader, classes = get_dataloaders("data")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10

for epoch in range(epochs):
    running_loss = 0.0
    model.train()

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch + 1} - Loss {running_loss:.4f}")
    torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
