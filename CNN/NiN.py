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
RESIZE=227
BATCH_SIZE = 64
LEARNING_RATE = 0.01
NUM_EPOCHS = 10
NUM_CLASSES = 10
NUM_WORKERS = 8

# CUDA
device = utils.check_CUDA_available()

class NiN(nn.Module):
    """
    In modern models, it is more common to see such a pattern:
    CONVOLUTION -> ACTIVATION -> POOLING.
    
    Net Architecture:
        Input    | 3x227x227
        C1       | 96x55x55     11x11 stride=4 (MLPConv Block 1)
        C1-mlp1  | 96x55x55     1x1 stride=1
        C1-mlp2  | 96x55x55     1x1 stride=1
        P2       | 96x27x27     3x3 stride=2 (Max Pool)
        C3       | 256x27x27    5x5 stride=1 padding=2 (MLPConv Block 2)
        C3-mlp1  | 256x27x27    1x1 stride=1
        C3-mlp2  | 256x27x27    1x1 stride=1
        P4       | 256x13x13    3x3 stride=2 (Max Pool)
        C5       | 384x13x13    3x3 stride=1 padding=1 (MLPConv Block 3)
        C5-mlp1  | 384x13x13    1x1 stride=1
        C5-mlp2  | 384x13x13    1x1 stride=1
        P6       | 384x6x6      3x3 stride=2 (Max Pool)
        Dropout  | 384x6x6      Rate = 0.5
        C7       | 1000x6x6     3x3 stride=1 padding=1 (Final Block)
        C7-mlp1  | 1000x6x6     1x1 stride=1
        C7-mlp2  | 1000x6x6     1x1 stride=1
        GAP      | 1000x1x1     Global Average Pooling
        Output   | 1000         Softmax
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            self._nin_block(1, 96, kernel_size=11, stride=4, padding=0), # 227x227 -> 55x55
            nn.MaxPool2d(kernel_size=3, stride=2), # 55x55 -> 27x27
            
            self._nin_block(96, 256, kernel_size=5, stride=1, padding=2), # 27x27 -> 27x27
            nn.MaxPool2d(kernel_size=3, stride=2), # 27x27 -> 13x13
            
            self._nin_block(256, 384, kernel_size=3, stride=1, padding=1), # 13x13 -> 13x13
            nn.MaxPool2d(kernel_size=3, stride=2), # 13x13 -> 6x6
            nn.Dropout(0.5),
            
            self._nin_block(384, NUM_CLASSES, kernel_size=3, stride=1, padding=1), # 6x6 -> 6x6
            nn.AdaptiveAvgPool2d((1, 1)), # 6x6 -> 1x1
            
            nn.Flatten()
        )
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _nin_block(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1), # 1st 1x1 conv
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1), # 2nd 1x1 conv
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)
    
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

def train_one_epoch(model:NiN|nn.Module, 
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

def validation(model:NiN|nn.Module,
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

def visualization(model:NiN|nn.Module,
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
    model = NiN()
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