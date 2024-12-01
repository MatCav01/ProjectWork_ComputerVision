from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch
from dataset_loading import CIM_Dataset
from nn_loading import MultiLabelResnet18, train_model, valid_model, test_model

batch_size = 64
learning_rate = 0.001
n_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = CIM_Dataset(root='./DATABASE_CIM', valid=False, transform=data_transform)
n_classes = len(dataset.classes)
with torch.Generator() as generator:
    generator.manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, lengths=[0.8, 0.2], generator=generator)
valid_dataset = CIM_Dataset(root='./DATABASE_CIM', valid=True, transform=data_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model_net = MultiLabelResnet18(n_classes).to(device)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(params=model_net.parameters(), lr=learning_rate)

# train and valid
for epoch in range(n_epochs):
    train_loss, train_acc = train_model(model_net, train_loader, criterion, optimizer, device, n_classes)
    valid_loss, valid_acc = valid_model(model_net, valid_loader, criterion, device, n_classes)
    print(f'Epoch {epoch + 1}/{n_epochs}:')
    print(f'\tTrain Loss: {train_loss:.6f}\n\tTrain Accuracy: {train_acc}')
    print(f'\tValid Loss: {valid_loss:.6f}\n\tValid Accuracy: {valid_acc}\n')
