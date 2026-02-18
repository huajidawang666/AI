import torch
from dataset.IMDB import load_IMDB_dataset
from torch import nn, optim
from utils.metric import Accumulator

# --- 4. 简单 RNN 模型 ---

class RNNLayer(nn.Module):
    def __init__(self, emb_dim, hid_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.Wx = nn.Linear(emb_dim, hid_dim)
        self.Wh = nn.Linear(hid_dim, hid_dim)
        self.activation = nn.Tanh()
        
    def forward(self, x, hidden=None):
        batch_size, seq_len, _ = x.size()
        outputs = [] # 
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hid_dim, device=x.device)
        for t in range(seq_len):
            hidden = self.activation(self.Wx(x[:, t, :]) + self.Wh(hidden))
            outputs.append(hidden.unsqueeze(1)) # (batch_size, 1, hid_dim)
        return torch.cat(outputs, dim=1), hidden

class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = RNNLayer(emb_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, 1)
        
    def forward(self, x):
        x = self.embedding(x)
        _, hidden = self.rnn(x)
        return self.fc(hidden)



# --- 5. 训练循环 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, test_loader, VOCAB_SIZE = load_IMDB_dataset()
model = SimpleRNN(VOCAB_SIZE, 100, 128).to(DEVICE)
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