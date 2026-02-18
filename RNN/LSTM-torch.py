import torch
from dataset.IMDB import load_IMDB_dataset
from torch import nn, optim
from utils.metric import Accumulator

# --- 4. 简单 LSTM 模型 ---

class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, batch_first=True)
        self.fc = nn.Linear(hid_dim, 1)
        
    def forward(self, x):
        x = self.embedding(x)
        _, _, hidden = self.rnn(x)
        return self.fc(hidden)



# --- 5. 训练循环 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, test_loader, VOCAB_SIZE = load_IMDB_dataset()
model = SimpleLSTM(VOCAB_SIZE, 100, 128).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

print(f"开始训练 (设备: {DEVICE})...")
for epoch in range(50):
    model.train()
    metric = Accumulator(2)  # 记录总损失和样本数
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
            metric.add(labels.size(0), acc)
    print(f"Epoch {epoch+1} 完成，Loss: {loss.item():.4f}, Accuracy: {metric[1]/metric[0]:.4f}")