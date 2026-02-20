import math
import torch
from dataset.IMDB_CLS import load_IMDB_dataset
from torch import device, nn, optim
from utils.metric import Accumulator
from RNN.Transformer import TransformerClassifier

# --- 5. 训练循环 ---

NUM_LAYERS = 6
D_MODEL = 512
D_FF = 2048
NUM_HEADS = 8
MAX_LEN = 200
EPOCHS = 50
WARMUP_RATIO = 0.1  # warmup steps portion of total training steps

DEVICE = device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, test_loader, VOCAB_SIZE = load_IMDB_dataset(batch_size=256, max_len=MAX_LEN)
model = TransformerClassifier(num_layers=NUM_LAYERS,
                              vocab_size=VOCAB_SIZE,
                              d_model=D_MODEL,
                              d_ff=D_FF,
                              num_heads=NUM_HEADS,
                              num_classes=1,
                              max_len=MAX_LEN).to(DEVICE)
torch.compile(model)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()


def build_warmup_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

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
total_steps = EPOCHS * len(train_loader)
warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
scheduler = build_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)
scaler = torch.amp.GradScaler('cuda')

for epoch in range(EPOCHS):
    model.train()
    metric = Accumulator(3)  # 记录总损失和样本数
    for texts, labels in train_loader:
        texts, labels = texts.to(DEVICE), labels.to(DEVICE).float()
        with torch.amp.autocast('cuda'):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(texts).squeeze(-1)
            loss = criterion(outputs, labels)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0) # 防止梯度爆炸
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
         
        with torch.no_grad():
            acc = ((outputs > 0) == (labels > 0.5)).sum().item()            
            metric.add(loss.item() * labels.size(0), acc, labels.size(0))
    test_acc = validation(model, test_loader)
            
    print(f"Epoch {epoch+1} 完成，Loss: {metric[0]/metric[2]:.4f}, Accuracy: {metric[1]/metric[2]:.4f}, Test Accuracy: {test_acc:.4f}")