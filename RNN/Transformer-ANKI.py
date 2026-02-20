import torch
import spacy
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


def build_idx_to_token(vocab: dict):
    idx_to_token = [None] * len(vocab)
    for tok, idx in vocab.items():
        if idx < len(idx_to_token):
            idx_to_token[idx] = tok
    return idx_to_token


def encode_en_sentence(sentence: str, en_vocab: dict, nlp, max_len: int, sos_idx=1, eos_idx=2, unk_idx=3, pad_idx=0):
    tokens = [tok.text.lower() for tok in nlp(sentence) if tok.text.strip()]
    ids = [sos_idx] + [en_vocab.get(t, unk_idx) for t in tokens][: max_len - 2] + [eos_idx]
    if len(ids) < max_len:
        ids += [pad_idx] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
        ids[-1] = eos_idx
    return torch.tensor(ids, dtype=torch.long)


def greedy_decode(model, src_ids, zh_vocab, device, max_len: int, sos_idx=1, eos_idx=2, pad_idx=0):
    model.eval()
    idx_to_token = build_idx_to_token(zh_vocab)
    tgt = torch.tensor([[sos_idx]], device=device)
    src = src_ids.unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(max_len - 1):
            logits = model(src, tgt, src_pad_idx=pad_idx, tgt_pad_idx=pad_idx)
            next_token = logits[0, -1].argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_token.unsqueeze(0)], dim=1)
            if next_token.item() == eos_idx:
                break
    pred_ids = tgt.squeeze(0).tolist()[1:]  # drop SOS
    tokens = [idx_to_token[i] for i in pred_ids if i not in (pad_idx, sos_idx, eos_idx) and idx_to_token[i] is not None]
    return ''.join(tokens)

if __name__ == "__main__":
    file_path = config.DATA_DIR / 'ANKI' / 'cmn-eng' / 'cmn.txt'
    try:
        loader, en_v, zh_v = get_dataloader(file_path, batch_size=1024)
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            nlp_en = spacy.load("en_core_web_sm")
        except OSError as e:
            raise OSError("请先安装 spaCy 英文模型: python -m spacy download en_core_web_sm") from e

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
            epoch_loss = 0.0
            
            for src_batch, tgt_batch in loader:
                src_batch = src_batch.to(DEVICE)
                tgt_batch = tgt_batch.to(DEVICE)

                # 准备 Teacher Forcing 输入输出
                tgt_in, tgt_out = shift_targets(tgt_batch, PAD_IDX)

                optimizer.zero_grad(set_to_none=True)
                
                # 混合精度加速
                with torch.amp.autocast('cuda'):
                    # 确保你的 Transformer model 内部处理了 Padding Mask 和 Sequence Mask
                    logits = model(src_batch, tgt_in, src_pad_idx=PAD_IDX, tgt_pad_idx=PAD_IDX)
                    
                    # 展平进行交叉熵计算
                    # logits: (batch * (seq_len-1), vocab_size)
                    # tgt_out: (batch * (seq_len-1))
                    loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

                # 反向传播
                scaler.scale(loss).backward()
                
                # 梯度裁剪防止梯度爆炸
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 通常 1.0 比较常用
                
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                
            if epoch % 5 == 0:
                torch.save(model.state_dict(), f"transformer_epoch_{epoch}.pt")

            avg_loss = epoch_loss / len(loader)
            print(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {avg_loss:.4f}")

        # 保存最终模型
        torch.save(model.state_dict(), "final_model.pt")

        # Quick translation demo
        demo_en = "I love machine learning"
        src_ids = encode_en_sentence(demo_en, en_v, nlp_en, MAX_LEN, sos_idx=1, eos_idx=2, unk_idx=3, pad_idx=PAD_IDX).to(DEVICE)
        zh_pred = greedy_decode(model, src_ids, zh_v, DEVICE, max_len=MAX_LEN, sos_idx=1, eos_idx=2, pad_idx=PAD_IDX)
        print(f"示例英文: {demo_en}")
        print(f"模型翻译: {zh_pred}")
        
        
        
    except Exception as e:
        print(f"加载数据时出错: {e}")
        

        
    