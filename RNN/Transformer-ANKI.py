import torch
import config
from dataset.cmn_eng import get_dataloader
from RNN.Transformer import Transformer
from torch import nn, optim

# Hyperparameters
NUM_LAYERS = 4
D_MODEL = 256
D_FF = 1024
NUM_HEADS = 4
DROPOUT = 0.1
EPOCHS = 10
LR = 3e-4
MAX_LEN = 256
PAD_IDX = 0


def shift_targets(tgt_batch, pad_idx=PAD_IDX):
    """
    Prepare decoder inputs/targets for teacher forcing.
    Args:
        tgt_batch: (batch, seq_len) with SOS at position 0 and EOS somewhere inside.
    Returns:
        tgt_in:  (batch, seq_len-1) shifted right (drops last token)
        tgt_out: (batch, seq_len-1) shifted left (drops first token)
    """
    tgt_in = tgt_batch[:, :-1]
    tgt_out = tgt_batch[:, 1:]
    return tgt_in, tgt_out

if __name__ == "__main__":
    file_path = config.DATA_DIR / 'data' / 'ANKI' / 'cmn.txt'
    try:
        loader, en_v, zh_v = get_dataloader(file_path)
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = Transformer(
            num_layers=NUM_LAYERS,
            src_vocab_size=len(en_v),
            tgt_vocab_size=len(zh_v),
            d_model=D_MODEL,
            d_ff=D_FF,
            num_heads=NUM_HEADS,
            max_len=MAX_LEN,
            dropout=DROPOUT,
            bi_embedded=False,
            full_embedded=False,
        ).to(DEVICE)

        optimizer = optim.AdamW(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        scaler = torch.amp.GradScaler('cuda')

        print(f"开始训练 (设备: {DEVICE}) ...")
        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_loss = 0.0
            total_tokens = 0

            for src_batch, tgt_batch in loader:
                src_batch = src_batch.to(DEVICE)
                tgt_batch = tgt_batch.to(DEVICE)

                tgt_in, tgt_out = shift_targets(tgt_batch, PAD_IDX)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    logits = model(src_batch, tgt_in, src_pad_idx=PAD_IDX, tgt_pad_idx=PAD_IDX)
                    vocab_size = logits.size(-1)
                    loss = criterion(logits.view(-1, vocab_size), tgt_out.reshape(-1))

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()

                non_pad = (tgt_out != PAD_IDX).sum().item()
                total_loss += loss.item() * max(non_pad, 1)
                total_tokens += max(non_pad, 1)

            avg_loss = total_loss / max(total_tokens, 1)
            print(f"Epoch {epoch}/{EPOCHS} - Loss: {avg_loss:.4f}")

        # Quick sample batch preview
        en_example, zh_example = next(iter(loader))
        print("英文示例索引:", en_example[:2])
        print("中文示例索引:", zh_example[:2])
    except Exception as e:
        print(f"加载数据时出错: {e}")
        

        
    