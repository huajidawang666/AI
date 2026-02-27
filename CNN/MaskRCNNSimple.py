import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader, random_split
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import v2 as T
from torchvision import tv_tensors
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import time

from dataset.MoNuSeg import MoNuSegDataset


# ──────────────────────────────────────────────────────────
# 1. 实例化 Dataset（复用上次写的包装器，加上数据增强）
# ──────────────────────────────────────────────────────────
class MoNuSegInstanceDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform=None, min_area=50):
        self.base = base_dataset
        self.transform = transform
        self.min_area = min_area

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, label = self.base[idx]
        image_float = image.float() / 255.0
        semantic = label.squeeze(0).numpy()

        inside = (semantic == 1).astype(np.uint8)
        num_instances, instance_map = cv2.connectedComponents(inside)

        instance_masks, boxes = [], []
        for inst_id in range(1, num_instances):
            mask = (instance_map == inst_id).astype(np.uint8)
            if mask.sum() < self.min_area:
                continue
            ys, xs = np.where(mask)
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            if x2 <= x1 or y2 <= y1:
                continue
            instance_masks.append(mask)
            boxes.append([float(x1), float(y1), float(x2), float(y2)])

        if len(boxes) == 0:
            return None

        H, W = semantic.shape

        target = {
            # ✅ 包装成 BoundingBoxes，指定格式和画布尺寸
            'boxes': tv_tensors.BoundingBoxes(
                torch.as_tensor(boxes, dtype=torch.float32),
                format='XYXY',
                canvas_size=(H, W)
            ),
            'labels': torch.ones(len(boxes), dtype=torch.int64),
            # ✅ 包装成 Mask，shape 必须是 [N, H, W]
            'masks': tv_tensors.Mask(
                torch.as_tensor(np.stack(instance_masks), dtype=torch.uint8)
            ),
        }

        if self.transform:
            image_float, target = self.transform(image_float, target)

        # 传给 Mask R-CNN 前要解包回普通 Tensor
        return image_float, {
            'boxes':  target['boxes'].data,
            'labels': target['labels'],
            'masks':  target['masks'].data,
        }


def get_transform(train=True):
    """简单但有效的数据增强"""
    transforms = []
    if train:
        transforms += [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            # 颜色抖动对 H&E 染色图像很有帮助
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        ]
    return T.Compose(transforms) if transforms else None


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return [], []
    images, targets = zip(*batch)
    return list(images), list(targets)


# ──────────────────────────────────────────────────────────
# 2. 构建模型（从头训练，anchor 针对小目标调整）
# ──────────────────────────────────────────────────────────
def build_model_from_scratch(num_classes=2):
    # backbone 不加载预训练权重
    backbone = resnet_fpn_backbone(
        backbone_name='resnet50',
        weights=None,           # 不用预训练
        trainable_layers=5,     # 全部层可训练
    )

    # 细胞核很小，anchor 尺寸要小
    # 默认是 (32,64,128,256,512)，这里改为更适合细胞的尺寸
    anchor_generator = AnchorGenerator(
        sizes=((8,), (16,), (32,), (64,), (128,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )

    # RoI Align 输出尺寸
    roi_pooler = torch.ops.torchvision.roi_align  # 直接用 MaskRCNN 默认即可

    model = MaskRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        # 针对密集小目标调整 RPN 参数
        rpn_pre_nms_top_n_train=4000,
        rpn_pre_nms_top_n_test=2000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        # 每张图采样更多 RoI，因为细胞核密集
        box_batch_size_per_image=256,
        box_detections_per_img=300,   # 一张 patch 可能有很多细胞核
        min_size=256,
        max_size=256,
    )

    return model


# ──────────────────────────────────────────────────────────
# 3. 训练函数
# ──────────────────────────────────────────────────────────
def train_one_epoch(model, optimizer, loader, device, epoch, print_freq=20):
    model.train()
    total_loss = 0
    loss_components = {}
    n_batches = 0

    for i, (images, targets) in enumerate(loader):
        if not images:
            continue

        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        # 梯度裁剪，从头训练时防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += losses.item()
        n_batches += 1

        # 累加各分量
        for k, v in loss_dict.items():
            loss_components[k] = loss_components.get(k, 0) + v.item()

        if (i + 1) % print_freq == 0:
            avg = total_loss / n_batches
            print(f"  Epoch [{epoch}] Step [{i+1}/{len(loader)}]  loss={avg:.4f}  "
                  + "  ".join(f"{k}={v/n_batches:.3f}" for k, v in loss_components.items()))

    avg_loss = total_loss / max(n_batches, 1)
    avg_components = {k: v / max(n_batches, 1) for k, v in loss_components.items()}
    return avg_loss, avg_components


@torch.no_grad()
def evaluate(model, loader, device):
    """简单用 val loss 来监控，不跑完整 mAP（快）"""
    model.train()  # 必须 train 模式才能拿到 loss
    total_loss = 0
    n_batches = 0

    for images, targets in loader:
        if not images:
            continue
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        total_loss += sum(loss_dict.values()).item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ──────────────────────────────────────────────────────────
# 4. 主训练流程
# ──────────────────────────────────────────────────────────
def main():
    # 超参数
    NUM_EPOCHS   = 50
    BATCH_SIZE   = 4
    LR           = 1e-3
    WEIGHT_DECAY = 1e-4
    VAL_RATIO    = 0.15
    SAVE_DIR     = Path("checkpoints")
    SAVE_DIR.mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 数据集
    base_dataset = MoNuSegDataset()
    train_base   = MoNuSegDataset(transform=None)  # 共享同一路径，transform 在包装层做

    full_dataset = MoNuSegInstanceDataset(base_dataset, min_area=50)
    n_val  = int(len(full_dataset) * VAL_RATIO)
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # 给 train split 单独套增强（通过重新包装）
    # 更干净的做法：直接在 __getitem__ 里判断是否 augment
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=2, shuffle=False,
        num_workers=2, collate_fn=collate_fn
    )
    print(f"Train: {n_train} patches, Val: {n_val} patches")

    # 模型
    model = build_model_from_scratch(num_classes=2)
    model.to(device)

    # 优化器：从头训练用较大 lr，配合 warmup
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)

    # ── Warmup：前5个epoch线性升lr ──
    def warmup_lambda(epoch):
        if epoch < 5:
            return (epoch + 1) / 5
        return 1.0
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)

    # 训练记录
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss, components = train_one_epoch(
            model, optimizer, train_loader, device, epoch
        )
        val_loss = evaluate(model, val_loader, device)

        # 学习率调度
        if epoch <= 5:
            warmup_scheduler.step()
        else:
            scheduler.step()

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch:3d}/{NUM_EPOCHS}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"lr={current_lr:.2e}  time={elapsed:.1f}s")
        print("  " + "  ".join(f"{k}={v:.3f}" for k, v in components.items()))

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, SAVE_DIR / 'best_model.pth')
            print(f"  ✓ 保存最佳模型 (val_loss={val_loss:.4f})")

        # 每10轮保存一次 checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, SAVE_DIR / f'checkpoint_epoch{epoch}.pth')

    # 绘制 loss 曲线
    plt.figure(figsize=(8, 4))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'],   label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('MaskRCNN on MoNuSeg - Training Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('loss_curve.png', dpi=150)
    plt.show()
    print("训练完成！Loss 曲线已保存到 loss_curve.png")


# ──────────────────────────────────────────────────────────
# 5. 加载训练好的模型做推理
# ──────────────────────────────────────────────────────────
def load_and_infer(checkpoint_path, dataset, indices=(0, 5, 10), score_thresh=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model_from_scratch(num_classes=2)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"加载模型：epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

    for idx in indices:
        sample = dataset[idx]
        if sample is None:
            continue
        image, gt_target = sample

        with torch.no_grad():
            pred = model([image.to(device)])[0]
        pred = {k: v.cpu() for k, v in pred.items()}

        _visualize(image, gt_target, pred, score_thresh)


def _visualize(image_tensor, gt_target, pred, score_thresh=0.5):
    img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    keep = pred['scores'] > score_thresh

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, title, boxes, masks, scores in [
        (axes[0], f"GT ({len(gt_target['boxes'])} cells)",
         gt_target['boxes'], gt_target['masks'], None),
        (axes[1], f"Pred ({keep.sum()} cells, thresh={score_thresh})",
         pred['boxes'][keep], pred['masks'][keep].squeeze(1), pred['scores'][keep]),
    ]:
        ax.imshow(img_np)
        overlay = np.zeros_like(img_np)
        for i, (box, mask) in enumerate(zip(boxes, masks)):
            color = np.random.rand(3)
            overlay[(mask.numpy() > 0.5)] = color
            x1, y1, x2, y2 = box.int().tolist()
            ax.add_patch(patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=1, edgecolor=color, facecolor='none'
            ))
            if scores is not None:
                ax.text(x1, y1-2, f"{scores[i]:.2f}", fontsize=6, color='white',
                        bbox=dict(facecolor=color, alpha=0.6, pad=1))
        ax.imshow(overlay, alpha=0.45)
        ax.set_title(title, fontsize=12)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

    # 训练完后推理：
    # base = MoNuSegDataset()
    # dataset = MoNuSegInstanceDataset(base, min_area=50)
    # load_and_infer("checkpoints/best_model.pth", dataset)