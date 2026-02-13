from config import DATA_DIR
from torch.utils.data import Dataset, dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import torch

class TinyImageNetDataset(Dataset):
    def __init__(self, root=None, train=True, resize=227, image_size=64, transform=None):
        self.train_path = root if root else DATA_DIR / 'Tiny-ImageNet' / 'train'
        self.valid_path = root if root else DATA_DIR / 'Tiny-ImageNet' / 'val'
        self.resize = resize
        self.dataset = []
        self.image_size = image_size
        self.transform = transform
        self.classes = sorted([dir for dir in self.train_path.iterdir() if dir.is_dir()])        
        self.classes_dict = {dir.name: i for i, dir in enumerate(self.classes)}
        self._process_train_dir() if train else self._process_val_dir()
        
    def _process_train_dir(self):
        for dir in self.classes:
            img_dir = self.train_path / dir / 'images'
            coords_dict = {}
            try:
                with open(self.train_path / dir.name / (dir.name + '_boxes.txt'), 'r') as f:
                    for line in f:
                        parts = line.split()
                        if not parts: continue
                        
                        filename = parts[0]
                        coords_dict[filename] = [int(x) / self.image_size for x in parts[1:]]
                        
                for image in img_dir.glob('*.JPEG'):
                    self.dataset.append({
                        'path': image,
                        'label': self.classes_dict[dir.name],
                        'coords': coords_dict[image.name]
                    })
            except Exception as e:
                print(f'[Error] Error occurred in loading Tiny-ImageNet Dataset: {e}')
        
    def _process_val_dir(self):
        img_dir = self.valid_path / 'images'
        try:
            with open(self.valid_path / 'val_annotations.txt', 'r') as f:
                for line in f:
                    parts = line.split()
                    if not parts: continue
                    
                    filename = parts[0]
                    label = self.classes_dict[parts[1]]
                    coords = [int(x) / self.image_size for x in parts[2:]]
                    
                    self.dataset.append({
                        'path': img_dir / filename,
                        'label': label,
                        'coords': coords
                    })
        except Exception as e:
            print(f'[Error] Error occurred in loading Tiny-ImageNet Dataset: {e}')

    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        img = Image.open(self.dataset[idx]['path'], 'r').convert('RGB')
        img = img.resize((self.resize, self.resize))
        img = np.asarray(img, dtype=float) / 255.0
        img = img.transpose((2, 0, 1))
        img_tensor = torch.from_numpy(img).type(dtype=torch.float32)
        img_tensor = self.transform(img_tensor) if self.transform is not None else img_tensor
        return img_tensor, self.dataset[idx]['label'], torch.Tensor(self.dataset[idx]['coords'])
    

if __name__ == '__main__':
    tinyImageNetDataset = TinyImageNetDataset()
    print(tinyImageNetDataset[0])