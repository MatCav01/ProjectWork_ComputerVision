from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from cim_dataset import CIM_Dataset, CIM_Dataset_Test
from train_nn import MultiLabelResNet, CIM_Net, train_model, valid_test_model

batch_size = 64
learning_rate = 0.001
n_epochs = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

train_dataset = CIM_Dataset(root='./DATABASE_CIM', valid=False, transform=data_transform)
valid_dataset = CIM_Dataset(root='./DATABASE_CIM', valid=True, transform=data_transform)
test_dataset = CIM_Dataset_Test(train_dataset, valid_dataset, train_amount=0.5, n_samples=512)

n_classes = len(train_dataset.classes)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model_net = MultiLabelResNet(n_classes, transfer_learning=True).to(device)
# model_net = CIM_Net(n_classes).to(device)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(params=model_net.parameters(), lr=learning_rate)

# train and valid
for epoch in range(n_epochs):
    train_loss, train_acc = train_model(model_net, train_loader, criterion, optimizer, device, n_classes)
    valid_loss, valid_acc = valid_test_model(model_net, valid_loader, criterion, device, n_classes, test=False)
    
    print(f'Epoch {epoch + 1}/{n_epochs}:')
    print(f'\tTrain Loss: {train_loss:.6f}\n\tTrain Accuracy: {train_acc}')
    print(f'\tValid Loss: {valid_loss:.6f}\n\tValid Accuracy: {valid_acc}\n')

# test
test_loss, test_acc, test_acc_macroAvg = valid_test_model(model_net, test_loader, criterion, device, n_classes, test=True)
print(f'Test Loss: {test_loss:.6f}\nTest Accuracy per class: {test_acc}\nCumulative Test Accuracy: {test_acc_macroAvg:.3f}')

save_weigths = input('\nWould you like to save weights and biases? ')
if save_weigths in ['y', 'yes', 'Y', 'Yes', 'YES']:
    torch.save(model_net.state_dict(), './weights_biases.pt')
    print('WEIGHTS AND BIASES SAVED!')
else:
    print('WEIGHTS AND BIASES NOT SAVED!')