from torchvision import models
import torch
from torchmetrics.classification.accuracy import MultilabelAccuracy

class MultiLabelResnet18(torch.nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = torch.nn.Sequential(
            torch.nn.Linear(self.resnet.fc.in_features, n_classes),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x):
        out = self.resnet(x)
        return out

def train_model(model, train_loader, criterion, optimizer, device, n_classes):
    model.train()
    train_loss = 0.0
    mla = MultilabelAccuracy(num_labels=n_classes, average=None)
    accuracy = [0.0 for _ in range(n_classes)]
    accuracy = torch.tensor(accuracy, device=device)
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        accuracy += mla(outputs, labels)

    train_loss /= len(train_loader)
    accuracy /= len(train_loader)

    return train_loss, accuracy.numpy()

def valid_model(model, valid_loader, criterion, device, n_classes):
    model.eval()
    valid_loss = 0.0
    mla = MultilabelAccuracy(num_labels=n_classes, average=None)
    accuracy = [0.0 for _ in range(n_classes)]
    accuracy = torch.tensor(accuracy, device=device)
    
    for images, labels in valid_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        valid_loss += loss.item()
        accuracy += mla(outputs, labels)

    valid_loss /= len(valid_loader)
    accuracy /= len(valid_loader)

    return valid_loss, accuracy.numpy()

def test_model(model, test_loader, criterion, device, n_classes):
    model.eval()
    test_loss = 0.0
    mla = MultilabelAccuracy(num_labels=n_classes, average=None)
    accuracy = [0.0 for _ in range(n_classes)]
    accuracy = torch.tensor(accuracy, device=device)
    
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        accuracy += mla(outputs, labels)

    test_loss /= len(test_loader)
    accuracy /= len(test_loader)

    return test_loss, accuracy.numpy()