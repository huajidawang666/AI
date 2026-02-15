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
RESIZE=32
BATCH_SIZE = 256
LEARNING_RATE = 1e-1
MIN_LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
NUM_EPOCHS = 100
NUM_CLASSES = 10
NUM_WORKERS = 8

AUX_DROPOUT = 0.7
FINAL_DROPOUT = 0.4
LOSS_WEIGHT_AUX = 0.3

# CUDA
device = None

def get_device():
    global device
    if device is None:
        device = utils.check_CUDA_available()
    return device

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    """
    In modern models, it is more common to see such a pattern:
    CONVOLUTION -> ACTIVATION -> POOLING.
    
    Net Architecture:
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        # stem
        # assume input as 3x32x32
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) # disable bias with BN following
                                                                                      # 32x32 -> 32x32
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # 32x32 -> 16x16
        
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1) # 16x16 -> 16x16
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2) # 16x16 -> 8x8
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2) # 8x8 -> 4x4
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # 4x4 -> 1x1
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256, num_classes)
        
        self._initialize_weights()
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
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
    return compare.type(dtype=targets.dtype).sum()

def train_one_epoch(model:ResNet18|nn.Module, 
                    dataloader:DataLoader, 
                    optimizer:torch.optim.Optimizer,
                    scaler:torch.amp.GradScaler):
    metric = Accumulator(3)
    model.train()
    for inputs, targets in tqdm.tqdm(dataloader, mininterval=2.0):
        # deduce type explicitly
        inputs:torch.Tensor
        targets:torch.Tensor

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # 训练循环中
        with torch.amp.autocast('cuda'):
            optimizer.zero_grad(set_to_none=True)
            predicts= model(inputs)
            loss = torch.nn.functional.cross_entropy(predicts, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        with torch.no_grad():
            acc = accuracy(predicts, targets)
            metric.add(loss.detach(), acc, targets.numel())
        
    # return loss and accuracy
    return metric[0]/metric[2], metric[1]/metric[2]

def validation(model:ResNet18|nn.Module,
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

def visualization(model:ResNet18|nn.Module,
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
    device = get_device()
    data_dir = config.DATA_DIR / 'CIFAR-10'
    train_transform = transforms.Compose([
        # transforms.Resize((RESIZE, RESIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(RESIZE, padding=4), # data enhancement
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    test_transform = transforms.Compose([
        # transforms.Resize((RESIZE, RESIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)
    
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
    model = ResNet18()
    model.to(device)
    model = torch.compile(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=MIN_LEARNING_RATE)
    
    scaler = torch.amp.GradScaler('cuda')
    
    # train
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc= train_one_epoch(model=model, 
                               dataloader=train_loader, 
                               optimizer=optimizer,
                               scaler=scaler)
        # val_acc = 0
        val_acc = validation(model=model,
                              dataloader=test_loader)
        print(f"""Epoch {epoch+1:>5} | train loss: {train_loss}, train acc: {train_acc}
              val acc: {val_acc}
              learning rate: {scheduler.get_last_lr()[0]}""")
        
        scheduler.step()
        
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