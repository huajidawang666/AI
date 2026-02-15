from time import time

import dataset
import torch
import utils
import tqdm
import config
from torchvision import datasets, transforms
from utils.metric import Accumulator
from torch.utils.data import DataLoader
from torch import nn

# Hyperparameters
RESIZE=224
BATCH_SIZE = 64
LEARNING_RATE = 0.01
NUM_EPOCHS = 10
NUM_CLASSES = 10
NUM_WORKERS = 8

# CUDA
device = utils.check_CUDA_available()

class VGG(nn.Module):
    """
    In modern models, it is more common to see such a pattern:
    CONVOLUTION -> ACTIVATION -> POOLING.
    
    Net Architecture:
    Input  | 3x224x224
        C1-1   | 64x224x224    3x3 stride=1 padding=1
        C1-2   | 64x224x224    3x3 stride=1 padding=1
        P2     | 64x112x112    2x2 stride=2 (Max Pool)
        C3-1   | 128x112x112   3x3 stride=1 padding=1
        C3-2   | 128x112x112   3x3 stride=1 padding=1
        P4     | 128x56x56     2x2 stride=2 (Max Pool)
        C5-1   | 256x56x56     3x3 stride=1 padding=1
        C5-2   | 256x56x56     3x3 stride=1 padding=1
        C5-3   | 256x56x56     3x3 stride=1 padding=1
        P6     | 256x28x28     2x2 stride=2 (Max Pool)
        C7-1   | 512x28x28     3x3 stride=1 padding=1
        C7-2   | 512x28x28     3x3 stride=1 padding=1
        C7-3   | 512x28x28     3x3 stride=1 padding=1
        P8     | 512x14x14     2x2 stride=2 (Max Pool)
        C9-1   | 512x14x14     3x3 stride=1 padding=1
        C9-2   | 512x14x14     3x3 stride=1 padding=1
        C9-3   | 512x14x14     3x3 stride=1 padding=1
        P10    | 512x7x7       2x2 stride=2 (Max Pool)
        F11    | 4096          Full Connect Layer
        F12    | 4096          Full Connect Layer
        Output | 1000          Full Connect Layer (Softmax)
    """
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            # Block 1: 2 convs, 64 filters
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 2 convs, 128 filters
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 3 convs, 256 filters
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 3 convs, 512 filters
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5: 3 convs, 512 filters
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1) # 展平为 (batch_size, 512*7*7)
        x = self.classifier(x)
        return x
    
def accuracy(predicts:torch.Tensor, 
             targets:torch.Tensor):
    """
    Get num of right predicts compared to targets
    
    :return:
        return num of right predicts compared to targets
    """
    # predicts shall be a tensor in (batch, clssification)
    # while targets shell be a tensor in (batch,)
    if len(predicts.shape) > 1 and predicts.shape[1] > 1: # assert shape and output dim
        predicts = predicts.argmax(dim=1)
    compare:torch.Tensor = predicts.type(dtype=targets.dtype) == targets # ensure dtype matches
    return float(compare.type(dtype=targets.dtype).sum())

def train_one_epoch(model:VGG|nn.Module, 
                    dataloader:DataLoader, 
                    criterion:nn.modules.loss._Loss, 
                    optimizer:torch.optim.Optimizer):
    metric = Accumulator(3)
    model.train()
    for inputs, targets in tqdm.tqdm(dataloader):
        # deduce type explicitly
        inputs:torch.Tensor
        targets:torch.Tensor

        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        predicts            = model(inputs)
        loss:torch.Tensor   = criterion(predicts, targets)
        loss.mean().backward()
        optimizer.step()
        
        acc = accuracy(predicts, targets)
        metric.add(float(loss.sum()), acc, targets.numel())
        
    # return loss and accuracy
    return metric[0]/metric[2], metric[1]/metric[2]

def validation(model:VGG|nn.Module,
               dataloader:DataLoader):
    model.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for inputs, targets in dataloader:
            # deduce type explicitly
            inputs:torch.Tensor
            targets:torch.Tensor
        
            inputs = inputs.to(device)
            targets = targets.to(device)

            predicts = model(inputs)
            metric.add(accuracy(predicts, targets), targets.numel())
    return metric[0]/metric[1]        

def visualization(model:VGG|nn.Module,
                  dataset:DataLoader):
    import matplotlib.pyplot as plt

    print("\n--- Visualizing Predictions ---")
    model.eval()
    
    checkout_batch_size = 10
    checkout_loader = DataLoader(dataset, batch_size=checkout_batch_size, shuffle=True)
    images, labels = next(iter(checkout_loader))
    
    with torch.no_grad():
        outputs = model(images.to(device))
        preds = outputs.argmax(dim=1).cpu()

    plt.figure(figsize=(12, 5))
    for i in range(checkout_batch_size):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        color = 'green' if preds[i] == labels[i] else 'red'
        plt.title(f"Pred: {preds[i]}\nActual: {labels[i]}", color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    data_dir = config.DATA_DIR / 'FashionMNIST'
    transform = transforms.Compose([
        transforms.Resize((RESIZE, RESIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_set = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)
    
    print(f"Train dataset size: {len(train_set)}")
    print(f"Test dataset size: {len(test_set)}")
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=NUM_WORKERS)
    for inputs, targets in test_loader:
        print(inputs.shape)
        print(targets.shape)
        break
    
    # Check data loading speed
    import time
    start_time = time.time()
    for i, (images, labels) in enumerate(train_loader):
        if i >= 100: break
        pass

    end_time = time.time()
    print(f"Estm. Data loading speed: {100 / (end_time - start_time):.2f} it/s")
    
    # model
    model = VGG()
    criterion = nn.CrossEntropyLoss(reduction='none') # do mean() manually
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model.to(device)
    
    # train
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc= train_one_epoch(model=model, 
                               dataloader=train_loader, 
                               criterion=criterion, 
                               optimizer=optimizer)
        # val_acc = 0
        val_acc = validation(model=model,
                              dataloader=test_loader)
        print(f"""Epoch {epoch+1:>5} | train loss: {train_loss}, train acc: {train_acc}
                      val acc: {val_acc}""")
        
    # checkout
    for inputs, targets in test_loader:
        # deduce type explicitly
        inputs:torch.Tensor
        targets:torch.Tensor
        predicts:torch.Tensor
        
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        predicts = model(inputs)
        if len(predicts.shape) > 1 and predicts.shape[1] > 1: # assert shape and output dim
            predicts = predicts.argmax(dim=1)
        print("Test:")
        print("Predicts:", predicts[8:])
        print("Targets: ", targets[8:])
        break
        
    # matplotlib
    visualization(model, test_set)

if __name__ == "__main__":
    main()