from torchvision import models
import torch
from torchmetrics.classification.accuracy import MultilabelAccuracy
import timm

class MultiLabelResNet(torch.nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        # self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet = timm.create_model('resnet14t.c3_in1k', pretrained=True)
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

    mla = MultilabelAccuracy(num_labels=n_classes, average=None).to(device)
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
    accuracy = (100 * accuracy / len(train_loader)).cpu().numpy()

    return train_loss, accuracy

def valid_test_model(model, data_loader, criterion, device, n_classes, test = False):
    model.eval()
    total_loss = 0.0
    
    mla = MultilabelAccuracy(num_labels=n_classes, average=None).to(device)
    accuracy = [0.0 for _ in range(n_classes)]
    accuracy = torch.tensor(accuracy, device=device)

    if test:
        mla_macroAvg = MultilabelAccuracy(num_labels=n_classes, average='macro').to(device)
        accuracy_macroAvg = torch.tensor(0.0, device=device)
    
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        accuracy += mla(outputs, labels)

        if test:
            accuracy_macroAvg += mla_macroAvg(outputs, labels)

    total_loss /= len(data_loader)
    accuracy = (100 * accuracy / len(data_loader)).cpu().numpy()

    if test:
        accuracy_macroAvg = (100 * accuracy_macroAvg / len(data_loader)).cpu().numpy()
        return total_loss, accuracy, accuracy_macroAvg
    
    return total_loss, accuracy
