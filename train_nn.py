from torchvision import models
import torch
from torchmetrics.classification.accuracy import MultilabelAccuracy
import timm

class MultiLabelResNet(torch.nn.Module):
    def __init__(self, n_classes, transfer_learning = False):
        super().__init__()
        # self.resnet = timm.create_model('resnet14t.c3_in1k', pretrained=True)
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = torch.nn.Sequential(
            torch.nn.Linear(self.resnet.fc.in_features, n_classes),
            torch.nn.Sigmoid()
        )

        if transfer_learning:
            for params in self.resnet.parameters():
                params.requires_grad = False
            for params in self.resnet.fc.parameters():
                params.requires_grad = True
    
    def forward(self, x):
        out = self.resnet(x)
        return out
    
class CIM_Net(torch.nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(32 * 64 * 64, n_classes),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.fc(out.view(out.size(0), -1))

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
    accuracy = (100 * accuracy / len(train_loader)).tolist()

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
    accuracy = (100 * accuracy / len(data_loader)).tolist()

    if test:
        accuracy_macroAvg = (100 * accuracy_macroAvg / len(data_loader)).tolist()
        return total_loss, accuracy, accuracy_macroAvg
    
    return total_loss, accuracy
