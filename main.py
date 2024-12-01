from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch
from dataset_loading import CIM_Dataset
from nn_loading import MultiLabelResnet18
import matplotlib.pyplot as plt

batch_size = 64
learning_rate = 0.001
n_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = CIM_Dataset(root='./DATABASE_CIM', valid=False, transform=data_transform)
with torch.Generator() as generator:
    generator.manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, lengths=[0.8, 0.2], generator=generator)
valid_dataset = CIM_Dataset(root='./DATABASE_CIM', valid=True, transform=data_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model_net = MultiLabelResnet18(n_classes=len(dataset.classes)).to(device)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(params=model_net.parameters(), lr=learning_rate)

# train and valid
for epoch in range(n_epochs):
    model_net.train()
