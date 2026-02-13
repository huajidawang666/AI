import dataset
import torch
import utils
import tqdm
from utils.metric import Accumulator
from torch.utils.data import DataLoader
from torch import nn

# Hyperparameters
RESIZE=227
BATCH_SIZE = 64
LEARNING_RATE = 0.2
NUM_EPOCHS = 10
NUM_CLASSES = 200
NUM_WORKERS = 16

# CUDA
device = utils.check_CUDA_available()

class AlexNet(nn.Module):
    """
    In modern models, it is more common to see such a pattern:
    CONVOLUTION -> ACTIVATION -> POOLING.
    
    Net Architecture:
        Input  | 3x227x227
        C1     | 96x55x55     11x11 stride=4
        P2     | 96x27x27     3x3 stride=2 (Max Pool)
        C3     | 256x27x27    5x5 stride=1 padding=2
        P4     | 256x13x13    3x3 stride=2 (Max Pool)
        C5     | 384x13x13    3x3 stride=1 padding=1
        C6     | 384x13x13    3x3 stride=1 padding=1
        C7     | 256x13x13    3x3 stride=1 padding=1
        P8     | 256x6x6      3x3 stride=2 (Max Pool)
        F9     | 4096         Full Connect Layer
        F10    | 4096         Full Connect Layer
        Output | 1000         Full Connect Layer (Softmax)
    """
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), # Overlapping Pooling
            
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=9216, out_features=4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=NUM_CLASSES)
        )
        
    def forward(self, X):
        X = self.feature_extractor(X)
        X = torch.flatten(X, 1)
        return self.classifier(X)
    
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

def train_one_epoch(model:AlexNet|nn.Module, 
                    dataloader:DataLoader, 
                    criterion:nn.modules.loss._Loss, 
                    optimizer:torch.optim.Optimizer):
    metric = Accumulator(3)
    model.train()
    for inputs, targets, _ in tqdm.tqdm(dataloader):
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

def validation(model:AlexNet|nn.Module,
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

def visualization(model:AlexNet|nn.Module,
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
    # dataloader
    tinyImageNetDataset = dataset.TinyImageNetDataset(train=True, resize=RESIZE)
    test_dataset = dataset.TinyImageNetDataset(train=False, resize=RESIZE)
    print(f"Train dataset size: {len(tinyImageNetDataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    train_loader = DataLoader(tinyImageNetDataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=NUM_WORKERS)
    for inputs, targets, target_coords in test_loader:
        print(inputs.shape)
        print(targets.shape)
        print(target_coords.shape)
        break
    
    # model
    model = AlexNet()
    criterion = nn.CrossEntropyLoss(reduction='none') # do mean() manually
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    
    model.to(device)
    
    # train
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc= train_one_epoch(model=model, 
                               dataloader=train_loader, 
                               criterion=criterion, 
                               optimizer=optimizer)
        val_acc = 0
        # val_acc = validation(model=model,
        #                       dataloader=test_loader)
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
    visualization(model, test_dataset)

if __name__ == "__main__":
    main()