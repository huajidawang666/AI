from pathlib import Path
from torchvision import datasets, transforms

BASE_DIR = Path(__file__).resolve().parent
data_path = BASE_DIR / '../data/MNIST'

def load_from_MNIST(resize:int = 28):
    r"""
    Return train and test dataloader for MNIST dataset.
    Normalization: Z-Score
    $ x' = (x - mean) / std
    Z-Score centers the data around 0 with a standard deviation of 1.
    In MNIST, mean = 0.1307 and std = 0.3081.
    """
    transform = transforms.Compose([
        transforms.Resize(size=resize),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)
    
    return train_dataset, test_dataset