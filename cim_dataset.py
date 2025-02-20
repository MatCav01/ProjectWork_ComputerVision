from torchvision.datasets import VisionDataset, ImageFolder
from torch.utils.data import ConcatDataset, Subset
import torch
import os, sys

class CIM_Dataset(VisionDataset):
    """
    CIM Dataset
    
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
            print(f'Path \"{self.root}\" does not exist!', file=sys.stderr)
            sys.exit(1)
        
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
        targets = torch.tensor(targets, dtype=torch.float32)

        return images, targets
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image, _ = self.images[index]
        target = self.targets[index]

        return image, target

class CIM_Dataset_Test(VisionDataset):
    """
    CIM_Dataset_Test

    Args:
        train_dataset: Training Set
        valid_dataset: Validation Set
        train_amount: The amount of training samples out of the total number of samples
        n_samples: The total number of samples that will form the CIM Test Set
    """
    def __init__(self, train_dataset: CIM_Dataset, valid_dataset: CIM_Dataset, train_amount = 0.5, n_samples = 512):
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.train_amount = train_amount
        self.n_samples = n_samples

        # self.denominations = train_dataset.denominations
        # self.rotations = train_dataset.rotations
        # self.classes = train_dataset.classes
        # self.class_to_idx = train_dataset.class_to_idx

        self.data = self.load_data()

    def load_data(self):
        generator = torch.Generator()
        generator.manual_seed(42)

        train_length = int(self.train_amount * self.n_samples)
        indices = torch.randint(low=0, high=len(self.train_dataset), size=(train_length,), generator=generator).tolist()
        train_subset = Subset(self.train_dataset, indices)

        valid_length = self.n_samples - train_length
        indices = torch.randint(low=0, high=len(self.valid_dataset), size=(valid_length,), generator=generator).tolist()
        valid_subset = Subset(self.valid_dataset, indices)

        dataset = ConcatDataset([train_subset, valid_subset])

        return dataset

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image, target = self.data[index]

        return image, target
