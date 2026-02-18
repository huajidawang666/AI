import torch
import config
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from utils.metric import Accumulator
import re
import tarfile
import os
import requests
from collections import Counter

# --- 1. 下载并解压数据 (纯手动实现) ---
DATA_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
DATA_DIR = config.DATA_DIR / "IMDB"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = config.DATA_DIR / "IMDB" / "aclImdb_v1.tar.gz"

if not os.path.exists(DATA_DIR/"aclImdb"):
    print("正在下载 IMDB 数据集...")
    r = requests.get(DATA_URL, stream=True)
    with open(DATA_PATH, 'wb') as f:
        f.write(r.raw.read())
    with tarfile.open(DATA_PATH, 'r:gz') as tar:
        tar.extractall(path=DATA_DIR)

# --- 2. 简单的分词和词汇表构建 ---
def tokenize(text):
    # 只保留字母，转小写
    return re.sub(r'[^a-zA-Z]', ' ', text.lower()).split()

def build_vocab(data_dir, max_size=20000):
    words = []
    for label in ['pos', 'neg']:
        path = os.path.join(data_dir, label)
        for fname in os.listdir(path)[:2000]: # 先取部分数据快速构建词表
            with open(os.path.join(path, fname), 'r', encoding='utf-8') as f:
                words.extend(tokenize(f.read()))
    
    counter = Counter(words)
    # 预留 0 给 padding, 1 给 unknown
    vocab = {word: i+2 for i, (word, _) in enumerate(counter.most_common(max_size))}
    return vocab

print("构建词汇表中...")
VOCAB = build_vocab(DATA_DIR / "aclImdb" / "train")
VOCAB_SIZE = len(VOCAB) + 2

# --- 3. 自定义 Dataset ---
class IMDBDataset(Dataset):
    def __init__(self, root_dir, split, vocab, max_len=200):
        self.data = []
        self.max_len = max_len
        self.vocab = vocab
        for label_val, label_name in enumerate(['neg', 'pos']):
            path = os.path.join(root_dir, split, label_name)
            for fname in os.listdir(path):
                with open(os.path.join(path, fname), 'r', encoding='utf-8') as f:
                    tokens = tokenize(f.read())
                    # 转索引
                    ids = [vocab.get(t, 1) for t in tokens][:max_len]
                    # Padding
                    if len(ids) < max_len:
                        ids += [0] * (max_len - len(ids))
                    self.data.append((torch.tensor(ids), torch.tensor(label_val, dtype=torch.float32)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

print("加载数据到内存中 (这可能需要一分钟)...")
train_ds = IMDBDataset(DATA_DIR / "aclImdb", "train", VOCAB)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

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