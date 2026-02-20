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
    file_path = config.DATA_DIR / 'ANKI' / 'cmn-eng' / 'cmn.txt'
    try:
        loader, en_v, zh_v = get_dataloader(file_path, batch_size=1024)
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

        # Quick sample batch preview
        en_example, zh_example = next(iter(loader))
        print("英文示例索引:", en_example[:2])
        print("中文示例索引:", zh_example[:2])
    except Exception as e:
        print(f"加载数据时出错: {e}")
        

        
    