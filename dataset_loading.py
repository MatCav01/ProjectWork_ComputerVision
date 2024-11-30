from torchvision.datasets import VisionDataset, ImageFolder
from torch.utils.data import ConcatDataset
from torchvision import transforms
import torch
import os
import matplotlib.pyplot as plt

class CIM_Dataset(ImageFolder):
    """CIM Dataset
    
    Args:
        root: Root directory
        valid: If True create Validation Set, otherwise Train/Test Set (need to be splitted for having Test)
        transform: A function/transform that takes in a PIL Image and returnes a transformed version
    """
    def __init__(self, root,  valid = False, transform = None):
        # super().__init__(root, transform=transform)
        # self.root = os.path.abspath(self.root)
        self.root = os.path.abspath(os.path.expanduser(root) if isinstance(root, str) else root)
        self.valid = valid
        self.transform = transform

        self.denominations = ['5', '10', '20', '50', '100', '200', '500', '1000']
        self.rotations = ['A', 'B', 'C', 'D']
        self.classes = self.denominations + self.rotations
        self.class_to_idx = {cls : i for i, cls in enumerate(self.classes)}

        self.samples, self.targets = self.load_data()

    def load_data(self):
        self.root += os.path.sep + ('ValidationSet' if self.valid else 'DataSet')
        if not os.path.exists(self.root):
            print(self.root)
            raise RuntimeError(f'Path "{self.root}" does not exist')
        
        datasets = []
        targets = []
        for denomination in self.denominations:
            for rotation in self.rotations:
                data_dir = self.root + os.path.sep + denomination + '_CIM_A' #+ os.path.sep + rotation
                datasets.append(ImageFolder(root=data_dir, transform=self.transform))

                target = []
                for cls in self.classes:
                    target.append(1 if cls == denomination or cls == rotation else 0)
                targets.append(target)
        
        images = ConcatDataset(datasets)
        targets = torch.tensor(targets, dtype=torch.float32)

        return images, targets
    
    def __len__(self):
        return super().__len__()
    
    def __getitem__(self, index):
        image, _ = super().__getitem__(index)
        target = self.targets[index]

        return image, target

if __name__ == '__main__':
    cim = CIM_Dataset(root='./ProjectWork_ComputerVision/DATABASE_CIM', transform=transforms.ToTensor())
    plt.imshow(cim.__getitem__(0)[0])
