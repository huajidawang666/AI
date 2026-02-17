import torch
import config
from torch import nn
from torch.utils.data import DataLoader
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
import time

# --- 1. 配置参数 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
MAX_LEN = 200  # 每条评论只取前200个单词
EPOCHS = 5

# --- 2. 数据准备 ---
tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

# 加载并构建词汇表
data_path = config.DATA_DIR / 'IMDB'
train_iter = IMDB(path=data_path, split='train')
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>", "<pad>"])
vocab.set_default_index(vocab["<unk>"])

# 数据处理管道
label_pipeline = lambda x: 1.0 if x == 2 else 0.0 # IMDB标签2是正向
text_pipeline = lambda x: vocab(tokenizer(x))[:MAX_LEN] # 截断长度

def collate_batch(batch):
    label_list, text_list = [], []
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
    
    labels = torch.tensor(label_list, dtype=torch.float32)
    # 对齐长度
    texts = pad_sequence(text_list, batch_first=True, padding_value=vocab["<pad>"])
    return texts.to(DEVICE), labels.to(DEVICE)

# 创建 DataLoader
train_iter, test_iter = IMDB() # 重新获取迭代器
train_dataloader = DataLoader(list(train_iter), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(list(test_iter), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

# --- 3. 定义模型 ---
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.RNN(emb_dim, hid_dim, batch_first=True)
        self.fc = nn.Linear(hid_dim, output_dim)
        
    def forward(self, text):
        embedded = self.embedding(text)
        # RNN 返回: output (所有步骤状态), hidden (最后一步状态)
        output, hidden = self.rnn(embedded)
        # 取最后一步的隐藏状态: hidden 形状为 [1, batch, hid_dim]
        return self.fc(hidden.squeeze(0))

model = SimpleRNN(len(vocab), EMBEDDING_DIM, HIDDEN_DIM, 1).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

# --- 4. 训练与评估函数 ---
def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    for texts, labels in dataloader:
        optimizer.zero_grad()
        predicted = model(texts).squeeze(1)
        loss = criterion(predicted, labels)
        loss.backward()
        optimizer.step()
        
        # 计算准确率
        acc = ((torch.sigmoid(predicted) > 0.5) == labels).sum().item()
        total_acc += acc
        total_count += labels.size(0)
    return total_acc / total_count

def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0
    with torch.no_grad():
        for texts, labels in dataloader:
            predicted = model(texts).squeeze(1)
            acc = ((torch.sigmoid(predicted) > 0.5) == labels).sum().item()
            total_acc += acc
            total_count += labels.size(0)
    return total_acc / total_count

# --- 5. 执行运行 ---
print(f"开始在 {DEVICE} 上训练...")
for epoch in range(1, EPOCHS + 1):
    start_time = time.time()
    train_acc = train(train_dataloader)
    test_acc = evaluate(test_dataloader)
    
    print(f'Epoch: {epoch} | 耗时: {time.time()-start_time:.1f}s')
    print(f'\t训练准确率: {train_acc*100:.2f}% | 测试准确率: {test_acc*100:.2f}%')