# from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset, ImageFolder
from torch.utils.data import ConcatDataset
import os

class CIM_Dataset(VisionDataset):
    """CIM Dataset
    
    Args:
        root: Root directory
        valid: If True create Validation Set, otherwise Train/Test Set (need to be splitted for having Test)
        transform: A function/transform that takes in a PIL Image and returnes a transformed version
        target_transform: A function/transform that takes in the target and transforms it
    """
    def __init__(self, root,  valid = False, transform = None, target_transform = None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.root = os.path.abspath(root)
        self.valid = valid
        self.denominations = ['5', '10', '20', '50', '100', '200', '500', '1000']
        self.rotations = ['A', 'B', 'C', 'D']

        # self.data, self.targets = self.load_data()

    def load_data(self):
        self.root += os.path.sep + ('ValidationSet' if self.valid else 'DataSet')
        if not os.path.exists(self.root):
            raise RuntimeError(f'Path "{self.root}" does not exists')
        
        for denomination in self.denominations:
            pass

if __name__ == '__main__':
    CIM_Dataset(root='.').load_data()