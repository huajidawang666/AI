import torch
from dataset.IMDB import load_IMDB_dataset
from torch import device, nn, optim
from utils.metric import Accumulator

# --- 4. 简单 LSTM 模型 ---

class LSTMLayer(nn.Module):
    def __init__(self, emb_dim, hid_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        
        self.Wx = nn.Linear(emb_dim, 4 * hid_dim)  # 输入到门的权重
        self.Wh = nn.Linear(hid_dim, 4 * hid_dim)  # 隐藏状态到门的权重
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
        with torch.no_grad():
            # 初始化 Wx 的 bias (也可以同时初始化 Wh 的 bias)
            # 这是为了让 f 拥有更大的初始值，而不是 f = sigmoid(0) = 0.5。
            # 我们希望模型不要一开始就选择忘记一半的信息。
            self.Wx.bias[self.hid_dim : 2 * self.hid_dim].fill_(1.0)
            self.Wh.bias[self.hid_dim : 2 * self.hid_dim].fill_(1.0)
        
    def forward(self, x, cell=None, hidden=None):
        batch_size, seq_len, _ = x.size()
        if cell is None: cell = torch.zeros(batch_size, self.hid_dim, device=x.device)  
        if hidden is None: hidden = torch.zeros(batch_size, self.hid_dim, device=x.device)
        cells = []
        hiddens = []
        for t in range(seq_len):
            gates = self.Wx(x[:, t, :]) + self.Wh(hidden)  # (batch_size, 4*hid_dim)
            i, f, o, g = gates.chunk(4, dim=1)  # i: Input Gate
                                                # f: Forget Gate
                                                # o: Output Gate
                                                # g: Cell Candidate
            i = self.sigmoid(i)
            f = self.sigmoid(f)
            o = self.sigmoid(o)
            g = self.tanh(g)
            
            cell = f * cell + i * g
            hidden = o * self.tanh(cell)
            
            cells.append(cell.unsqueeze(1))  # (batch_size, 1, hid_dim)
            hiddens.append(hidden.unsqueeze(1))  # (batch_size, 1, hid_dim)
        return torch.cat(cells, dim=1), torch.cat(hiddens, dim=1), hidden # use hidden for final classification

class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = LSTMLayer(emb_dim, hid_dim)
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