import torch
import config
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import jieba
import spacy
from collections import Counter

# --- 1. 数据预处理类 ---
class TranslationDataset(Dataset):
    def __init__(self, file_path, max_vocab_size=50000):
        self.en_sentences = []
        self.zh_sentences = []

        try:
            self.spacy_en = spacy.load("en_core_web_sm")
        except OSError:
            raise OSError("spaCy model 'en_core_web_sm' not found. Install it via: python -m spacy download en_core_web_sm")
        
        # 特殊 Token
        self.PAD_IDX = 0
        self.SOS_IDX = 1
        self.EOS_IDX = 2
        self.UNK_IDX = 3
        
        # 读取数据并分词
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    # 英文用 spaCy，中文用 jieba
                    en_toks = [tok.text.lower() for tok in self.spacy_en(parts[0]) if tok.text.strip()]
                    zh_toks = list(jieba.cut(parts[1]))
                    self.en_sentences.append(en_toks)
                    self.zh_sentences.append(zh_toks)

        # 构建词表
        self.en_vocab = self._build_vocab(self.en_sentences, max_vocab_size)
        self.zh_vocab = self._build_vocab(self.zh_sentences, max_vocab_size)

    def _build_vocab(self, sentences, max_size):
        counter = Counter([word for sent in sentences for word in sent])
        vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        for word, _ in counter.most_common(max_size - 4):
            vocab[word] = len(vocab)
        return vocab

    def _numericalize(self, tokens, vocab):
        return [vocab.get(tok, self.UNK_IDX) for tok in tokens]

    def __len__(self):
        return len(self.en_sentences)

    def __getitem__(self, index):
        en_ids = [self.SOS_IDX] + self._numericalize(self.en_sentences[index], self.en_vocab) + [self.EOS_IDX]
        zh_ids = [self.SOS_IDX] + self._numericalize(self.zh_sentences[index], self.zh_vocab) + [self.EOS_IDX]
        return torch.tensor(en_ids), torch.tensor(zh_ids)

# --- 2. 动态 Padding (Collate Function) ---
def collate_fn(batch):
    """
    因为 Transformer 输入长度必须一致，我们在 Batch 级别进行动态 Padding
    """
    en_batch, zh_batch = zip(*batch)
    en_pad = pad_sequence(en_batch, batch_first=True, padding_value=0)
    zh_pad = pad_sequence(zh_batch, batch_first=True, padding_value=0)
    return en_pad, zh_pad

# --- 3. 实例化与加载 ---
def get_dataloader(file_path, batch_size=32):
    dataset = TranslationDataset(file_path)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    return dataloader, dataset.en_vocab, dataset.zh_vocab

# --- 4. 测试运行 ---
if __name__ == "__main__":
    # 假设你已经下载了 cmn.txt
    file_path = config.DATA_DIR / 'ANKI' / 'cmn-eng' / 'cmn.txt'
    try:
        loader, en_v, zh_v = get_dataloader(file_path)
        en_example, zh_example = next(iter(loader))
        
        print(f"源语言 Batch 形状: {en_example.shape}") # [Batch, Seq_Len]
        print(f"目标语言 Batch 形状: {zh_example.shape}")
        print(f"词表大小: EN={len(en_v)}, ZH={len(zh_v)}")
    except FileNotFoundError:
        print("请先确保目录下有 cmn.txt 文件")