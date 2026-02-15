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
LEARNING_RATE = 1e-3
MIN_LEARNING_RATE = 1e-5
NUM_EPOCHS = 100
NUM_CLASSES = 10
NUM_WORKERS = 8

AUX_DROPOUT = 0.7
FINAL_DROPOUT = 0.4
LOSS_WEIGHT_AUX = 0.3

# CUDA
device = utils.check_CUDA_available()

class InceptionBlock(nn.Module):
    """
    Inception Block Architecture:
    Input  | in_channels x H x W
    [   Branch1 | out_channels1 x H x W     1x1 conv
        Branch2 | out_channels3red x H x W  1x1 conv
                | out_channels3 x H x W     3x3 conv padding=1
        Branch3 | out_channels5red x H x W  1x1 conv
                | out_channels5 x H x W     5x5 conv padding=2
        Branch4 | out_channels_pool x H x W 3x3 max pool stride=1 padding=1
                | out_channels_pool x H x W 1x1 conv]
    
    """
    def __init__(self, in_channels, out_channels1, out_channels3red, out_channels3, out_channels5red, out_channels5, out_channels_pool):
        super().__init__()
        def conv_block(in_channels, out_channels, kernel_size, stride=1, padding=0):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        self.branch1 = conv_block(in_channels, out_channels1, kernel_size=1)
        
        self.branch2 = nn.Sequential(
            conv_block(in_channels, out_channels3red, kernel_size=1),
            conv_block(out_channels3red, out_channels3, kernel_size=3, padding=1),
        )
        
        self.branch3 = nn.Sequential(
            conv_block(in_channels, out_channels5red, kernel_size=1),
            conv_block(out_channels5red, out_channels5, kernel_size=5, padding=2),
        )
        
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            conv_block(in_channels, out_channels_pool, kernel_size=1),
        )
        
    def forward(self, x):
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        branch3_out = self.branch3(x)
        branch4_out = self.branch4(x)
        
        return torch.cat([branch1_out, branch2_out, branch3_out, branch4_out], dim=1)

class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # assume input as 16x16
        self.avg_pool = nn.AvgPool2d(kernel_size=5, stride=3, padding=2)                           # ix16x16 -> ix6x6
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1), nn.ReLU(inplace=True),          # ix5x5 -> 128x6x6
        )
        self.flatten = nn.Flatten()                                                     # 128x6x6 -> 4608
        self.fc1 = nn.Sequential(
            nn.Linear(128*6*6, 1024), nn.ReLU(inplace=True), nn.Dropout(AUX_DROPOUT)    # 4608 -> 1024
        )
        self.fc2 = nn.Linear(1024, num_classes)                                         # 1024 -> num_classes
    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class GoogleNet(nn.Module):
    """
    In modern models, it is more common to see such a pattern:
    CONVOLUTION -> ACTIVATION -> POOLING.
    
    Net Architecture:
    """
    def __init__(self, num_class=NUM_CLASSES, use_auxiliary=True):
        super().__init__()
        self.use_auxiliary = use_auxiliary
        
        # assume input as 3x32x32
        self.before_inception = self._get_before_inception()                    # 3x332x32 -> 192x32x32
        self.inception_3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)        # 192x32x32 -> 256x32x32
        self.inception_3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)      # 258x32x32 -> 480x32x32
        self.max_pool_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)      # 480x32x32 -> 480x16x16
        
        self.inception_4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)       # 480x16x16 -> 512x16x16
        self.inception_4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)      # 512x16x16 -> 512x16x16
        self.inception_4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)      # 512x16x16 -> 512x16x16
        self.inception_4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)      # 512x16x16 -> 528x16x16
        self.inception_4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)    # 528x16x16 -> 832x16x16
        self.max_pool_3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)      # 832x16x16 -> 832x8x8
        
        self.inception_5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)    # 832x8x8 -> 832x8x8
        self.inception_5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)    # 832x8x8 -> 1024x8x8
        self.final_avg_pool = nn.AdaptiveAvgPool2d((1, 1))                      # 1024x8x8 -> 1024x1x1
        self.final_flatten = nn.Flatten()                                       # 1024x1x1 -> 1024
        self.final_dropout = nn.Dropout(FINAL_DROPOUT)                          # 1024 -> 1024
        self.final_FC = nn.Linear(1024, num_class)                              # 1024 -> num_class
        
        if use_auxiliary:
            self.aux_classifier_1 = AuxiliaryClassifier(512, num_class) # after inception_4a
            self.aux_classifier_2 = AuxiliaryClassifier(528, num_class) # after inception_4d
        self._initialize_weights()
    
    def forward(self, x):
        x = self.before_inception(x)
        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.max_pool_2(x)
        
        x = self.inception_4a(x)
        aux_out1 = self.aux_classifier_1(x) if self.use_auxiliary and self.training else None
        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)
        aux_out2 = self.aux_classifier_2(x) if self.use_auxiliary and self.training else None
        x = self.inception_4e(x)
        x = self.max_pool_3(x)
        
        x = self.inception_5a(x)
        x = self.inception_5b(x)
        x = self.final_avg_pool(x)
        x = self.final_flatten(x)
        x = self.final_dropout(x)
        final_out = self.final_FC(x)

        if self.use_auxiliary and self.training:
            return final_out, aux_out1, aux_out2
        else:
            return final_out
        
        
    def _get_before_inception(self):
        # assume input as 32x32
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), # 32x32 -> 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1), # 32x32 -> 32x32
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1) # 32x32 -> 32x32
        )
    
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
    return float(compare.type(dtype=targets.dtype).sum())

def train_one_epoch(model:GoogleNet|nn.Module, 
                    dataloader:DataLoader, 
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
        predicts, aux_preds_1, aux_preds_2 = model(inputs)
        loss_1:torch.Tensor = torch.nn.functional.cross_entropy(predicts, targets, reduction='none') # do mean() manually
        loss_2:torch.Tensor = torch.nn.functional.cross_entropy(aux_preds_1, targets, reduction='none')
        loss_3:torch.Tensor = torch.nn.functional.cross_entropy(aux_preds_2, targets, reduction='none')
        loss = loss_1 + LOSS_WEIGHT_AUX * (loss_2 + loss_3)                 
        loss.mean().backward()
        optimizer.step()
        
        acc = accuracy(predicts, targets)
        metric.add(float(loss.sum()), acc, targets.numel())
        
    # return loss and accuracy
    return metric[0]/metric[2], metric[1]/metric[2]

def validation(model:GoogleNet|nn.Module,
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

def visualization(model:GoogleNet|nn.Module,
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
    model = GoogleNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=MIN_LEARNING_RATE)
    
    model.to(device)
    
    # train
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc= train_one_epoch(model=model, 
                               dataloader=train_loader, 
                               optimizer=optimizer)
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