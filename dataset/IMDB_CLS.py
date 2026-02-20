import torch
import config
import re
import tarfile
import os
import requests
from torch.utils.data import DataLoader, Dataset
from collections import Counter


# Special token indices
PAD_IDX = 0
UNK_IDX = 1
CLS_IDX = 2


DATA_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
DATA_DIR = config.DATA_DIR / "IMDB"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = config.DATA_DIR / "IMDB" / "aclImdb_v1.tar.gz"

def _download_IMDB_dataset():
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
    # 预留 0 给 padding, 1 给 unknown, 2 给 [CLS]
    offset = 3
    vocab = {word: i + offset for i, (word, _) in enumerate(counter.most_common(max_size))}
    return vocab

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
                    # 保留空间给 [CLS]
                    tokens = tokens[: max_len - 1]
                    # 转索引并在开头加入 [CLS]
                    ids = [CLS_IDX] + [vocab.get(t, UNK_IDX) for t in tokens]
                    # Padding
                    if len(ids) < max_len:
                        ids += [PAD_IDX] * (max_len - len(ids))
                    self.data.append((torch.tensor(ids), torch.tensor(label_val, dtype=torch.float32)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_IMDB_dataset(batch_size=256, max_len=200):
    _download_IMDB_dataset()
    
    print("构建词汇表中...")
    VOCAB = build_vocab(DATA_DIR / "aclImdb" / "train")
    VOCAB_SIZE = len(VOCAB) + 3  # PAD, UNK, CLS
        
    print("加载数据到内存中 (这可能需要一分钟)...")
    train_ds = IMDBDataset(DATA_DIR / "aclImdb", "train", VOCAB, max_len)
    test_ds = IMDBDataset(DATA_DIR / "aclImdb", "test", VOCAB, max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, VOCAB_SIZE