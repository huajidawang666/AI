import dataset
import torch
import utils
from utils.metric import Accumulator
from torch.utils.data import DataLoader
from torch import nn

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.2
NUM_EPOCHS = 50

# CUDA
device = utils.check_CUDA_available()

class LeNet5(nn.Module):
    """
    Net Architecture:
        Input  | 1x32x32
        C1     | 6x28x28     5x5 stride=1
        P2     | 6x14x14     2x2
        C3     | 16x10x10    5x5 stride=1
        P4     | 16x5x5      2x2
        C5     | 120x1x1     5x5 stride=1    Equivalent of a Full Connect Layer
        F6     | 84
        Output | 10
    """
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2), nn.Sigmoid(),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2), nn.Sigmoid(),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1), nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(   
            nn.Linear(in_features=120, out_features=84), nn.Sigmoid(),
            nn.Linear(in_features=84, out_features=10)
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
    # predicts shall be a tensor in (batch, 10)
    # while targets shell be a tensor in (batch,)
    if len(predicts.shape) > 1 and predicts.shape[1] > 1: # assert shape and output dim
        predicts = predicts.argmax(dim=1)
    compare:torch.Tensor = predicts.type(dtype=targets.dtype) == targets # ensure dtype matches
    return float(compare.type(dtype=targets.dtype).sum())

def train_one_epoch(model:LeNet5|nn.Module, 
                    dataloader:DataLoader, 
                    criterion:nn.modules.loss._Loss, 
                    optimizer:torch.optim.Optimizer):
    metric = Accumulator(3)
    model.train()
    for inputs, targets in dataloader:
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

def validation(model:LeNet5|nn.Module,
               dataloader:DataLoader):
    model.eval()
    metric = Accumulator(2)
    for inputs, targets in dataloader:
        # deduce type explicitly
        inputs:torch.Tensor
        targets:torch.Tensor
        
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        predicts = model(inputs)
        metric.add(accuracy(predicts, targets), targets.numel())
    return metric[0]/metric[1]        

def main():
    # dataloader
    train_dataset, test_dataset = dataset.load_from_MNIST(resize=32) # input feature of LeNet-5 is 32x32
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # model
    model = LeNet5()
    criterion = nn.CrossEntropyLoss(reduction='none') # do mean() manually
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    
    model.to(device)
    
    # train
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc= train_one_epoch(model=model, 
                               dataloader=train_loader, 
                               criterion=criterion, 
                               optimizer=optimizer)
        val_acc = validation(model=model,
                              dataloader=test_loader)
        print(f"""Epoch {epoch+1:>5} | train loss: {train_loss}, train acc: {train_acc}
                      val acc: {val_acc}""")

if __name__ == "__main__":
    main()
