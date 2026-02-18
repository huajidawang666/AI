import torch
from dataset.IMDB import load_IMDB_dataset
from torch import nn, optim
from utils.metric import Accumulator

# --- 4. 简单 LSTM 模型 ---
# HyperParameters
DROPOUT = 0.5

class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers=2, 
                            batch_first=True, bidirectional=True, dropout=DROPOUT)
        self.fc = nn.Linear(hid_dim * 2, 1)
        self.dropout = nn.Dropout(DROPOUT)
        
    def forward(self, x):
        x = self.dropout(self.embedding(x))
        _, (hidden, _) = self.lstm(x)
        cat_hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1) 
        return self.fc(self.dropout(cat_hidden))



# --- 5. 训练循环 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, test_loader, VOCAB_SIZE = load_IMDB_dataset()
model = SimpleLSTM(VOCAB_SIZE, 100, 128).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

def validation(model, dataloader):
    model.eval()
    metric = Accumulator(2)  # 记录总损失和正确预测数
    with torch.no_grad():
        for texts, labels in dataloader:
            texts, labels = texts.to(DEVICE), labels.to(DEVICE)
            outputs = model(texts).squeeze(1)
            acc = ((outputs > 0) == (labels > 0.5)).sum().item()
            metric.add(acc, labels.size(0))
    return metric[0] / metric[1]

print(f"开始训练 (设备: {DEVICE})...")
for epoch in range(50):
    model.train()
    metric = Accumulator(3)  # 记录总损失和样本数
    for texts, labels in train_loader:
        texts, labels = texts.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(texts).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0) # 防止梯度爆炸
        optimizer.step()        
        with torch.no_grad():
            acc = ((outputs > 0) == (labels > 0.5)).sum().item()            
            metric.add(loss.item() * labels.size(0), acc, labels.size(0))
    test_acc = validation(model, test_loader)
            
    print(f"Epoch {epoch+1} 完成，Loss: {metric[0]/metric[2]:.4f}, Accuracy: {metric[1]/metric[2]:.4f}, Test Accuracy: {test_acc:.4f}")