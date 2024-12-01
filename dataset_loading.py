from torchvision.datasets import VisionDataset, ImageFolder
from torch.utils.data import ConcatDataset
import torch
import os

class CIM_Dataset(VisionDataset):
    """CIM Dataset
    
    Args:
        root: Root directory
        valid: If True create Validation Set, otherwise Train/Test Set (needs to be splitted for having Test)
        transform: A function/transform that takes in a PIL Image and returnes a transformed version
    """
    def __init__(self, root,  valid = False, transform = None):
        super().__init__(root, transform=transform)
        self.root = os.path.abspath(self.root)
        self.valid = valid

        self.denominations = ['5', '10', '20', '50', '100', '200', '500', '1000']
        self.rotations = ['A', 'B', 'C', 'D']
        self.classes = self.denominations + self.rotations
        # self.class_to_idx = {cls : i for i, cls in enumerate(self.classes)}

        self.images, self.targets = self.load_data()

    def load_data(self):
        self.root = os.path.join(self.root, 'ValidationSet' if self.valid else 'DataSet')
        if not os.path.exists(self.root):
            print(self.root)
            raise RuntimeError(f'Path "{self.root}" does not exist')
        
        datasets = []
        targets = []
        for denomination in self.denominations:
            data_dir = os.path.join(self.root, f'{denomination}_CIM_A')
            image_folder = ImageFolder(root=data_dir, transform=self.transform)
            datasets.append(image_folder)

            for rotation_index in image_folder.targets:
                target = []
                for cls in self.classes:
                    target.append(1 if cls == denomination or cls == self.rotations[rotation_index] else 0)
                targets.append(target)
        
        images = ConcatDataset(datasets)
        targets = torch.tensor(targets)

        return images, targets
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image, _ = self.images[index]
        target = self.targets[index]

        return image, target
