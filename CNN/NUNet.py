"""
U-Net++ (Nested U-Net) implementation in PyTorch.

Key features:
- Nested skip connections with dense semantic aggregation.
- Optional deep supervision (returns multi-scale outputs).
- Configurable encoder width and normalization.

Reference:
Zhou et al., "UNet++: A Nested U-Net Architecture for Medical Image Segmentation" (2018).
"""

from typing import Tuple, Union
from torchvision.transforms import v2
from tqdm import tqdm
from dataset.MoNuSeg import MoNuSegDataset
from utils.metric import Accumulator
import torch
import config
import cv2
from torch import nn
from datetime import datetime
import torch.nn.functional as F

NUM_EPOCHS = 50

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str = "bn"):
        super().__init__()
        norm_layer = self._norm_layer(num_features=out_ch, norm=norm)
        norm_layer2 = self._norm_layer(num_features=out_ch, norm=norm)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=norm_layer is None),
            norm_layer if norm_layer else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=norm_layer2 is None),
            norm_layer2 if norm_layer2 else nn.Identity(),
            nn.ReLU(inplace=True),
        )

    def _norm_layer(self, num_features: int, norm: str):
        if norm == "bn":
            return nn.BatchNorm2d(num_features)
        if norm == "gn":
            return nn.GroupNorm(num_groups=8, num_channels=num_features)
        if norm is None or norm == "none":
            return None
        raise ValueError(f"Unsupported norm type: {norm}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class NestedUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        deep_supervision: bool = False,
        filters: Tuple[int, int, int, int, int] = (64, 128, 256, 512, 1024),
        norm: str = "bn",
    ):
        """
        Args:
            in_channels: Input image channels.
            num_classes: Output channels/classes.
            deep_supervision: If True, returns tuple of multi-scale outputs.
            filters: Channel sizes per stage.
            norm: Normalization type ("bn", "gn", or "none").
        """
        super().__init__()
        self.deep_supervision = deep_supervision

        f0, f1, f2, f3, f4 = filters

        # Encoder pathway (level 0 depth 0..4)
        self.conv0_0 = ConvBlock(in_channels, f0, norm)
        self.conv1_0 = ConvBlock(f0, f1, norm)
        self.conv2_0 = ConvBlock(f1, f2, norm)
        self.conv3_0 = ConvBlock(f2, f3, norm)
        self.conv4_0 = ConvBlock(f3, f4, norm)

        # Decoder / nested blocks
        self.conv0_1 = ConvBlock(f0 + f1, f0, norm)
        self.conv1_1 = ConvBlock(f1 + f2, f1, norm)
        self.conv2_1 = ConvBlock(f2 + f3, f2, norm)
        self.conv3_1 = ConvBlock(f3 + f4, f3, norm)

        self.conv0_2 = ConvBlock(f0 * 2 + f1, f0, norm)
        self.conv1_2 = ConvBlock(f1 * 2 + f2, f1, norm)
        self.conv2_2 = ConvBlock(f2 * 2 + f3, f2, norm)

        self.conv0_3 = ConvBlock(f0 * 3 + f1, f0, norm)
        self.conv1_3 = ConvBlock(f1 * 3 + f2, f1, norm)

        self.conv0_4 = ConvBlock(f0 * 4 + f1, f0, norm)

        self.final1 = nn.Conv2d(f0, num_classes, kernel_size=1)
        self.final2 = nn.Conv2d(f0, num_classes, kernel_size=1)
        self.final3 = nn.Conv2d(f0, num_classes, kernel_size=1)
        self.final4 = nn.Conv2d(f0, num_classes, kernel_size=1)

        self.pool = nn.MaxPool2d(2, 2)
        self._init_weights()
        
    def _up(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=target.shape[2:], mode="bilinear", align_corners=False)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        # Encoder
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # Nested decoder paths
        x0_1 = self.conv0_1(torch.cat([x0_0, self._up(x1_0, x0_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self._up(x2_0, x1_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self._up(x3_0, x2_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self._up(x4_0, x3_0)], dim=1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self._up(x1_1, x0_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self._up(x2_1, x1_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self._up(x3_1, x2_0)], dim=1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self._up(x1_2, x0_0)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self._up(x2_2, x1_0)], dim=1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self._up(x1_3, x0_0)], dim=1))

        if self.deep_supervision:
            out1 = self.final1(x0_1)
            out2 = self.final2(x0_2)
            out3 = self.final3(x0_3)
            out4 = self.final4(x0_4)
            return out1, out2, out3, out4

        return self.final4(x0_4)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: [N, C, H, W] 模型的原始输出
        targets: [N, H, W] 或 [N, C, H, W] 的 one-hot 标签
        """
        probs = F.softmax(logits, dim=1)
        
        if targets.dim() == 3:
            targets = F.one_hot(targets, num_classes=logits.size(1)).permute(0, 3, 1, 2).float()
        
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets, dims)
        cardinality = torch.sum(probs + targets, dims)
        
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        
        return 1 - dice_score.mean()


sync_transforms = v2.Compose([
    v2.RandomRotation(degrees=15),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomHorizontalFlip(p=0.5),
])

image_transforms = v2.Compose([
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_dataloader(batch_size: int = 24):
    dataset = MoNuSegDataset()
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=8, batch_size=batch_size, shuffle=True, pin_memory=True)
    return dataloader

dataloader = get_dataloader()
# quick test
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = NestedUNet(in_channels=3, num_classes=3, deep_supervision=False, norm='gn')
    model.to(device)
    torch.compile(model)
    
    weights = torch.tensor([5.0, 2.0, 1.0]).to(device)
    criterion_CE = nn.CrossEntropyLoss(weight=weights)
    criterion_Dice = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = torch.amp.GradScaler('cuda')
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        metric = Accumulator(3)
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)
            images, labels = sync_transforms(images, labels)
            images = image_transforms(images)
            labels = labels.squeeze(1).long()  # Convert (B, 1, H, W) to (B, H, W) for loss calculation
            
            optimizer.zero_grad()
            # with torch.amp.autocast('cuda'):
            # Assuming labels are already class indices
            outputs = model(images)
            if isinstance(outputs, tuple):
                # out: (B, 3, H, W)
                # labels: (B, H, W)
                loss_ce = sum(criterion_CE(out, labels) for out in outputs) / len(outputs)
                loss_dice = sum(criterion_Dice(out, labels) for out in outputs) / len(outputs)
                loss = loss_ce + loss_dice
                
            else:
                loss_ce = criterion_CE(outputs, labels)
                loss_dice = criterion_Dice(outputs, labels)
                loss = loss_ce + loss_dice

            # scaler.scale(loss).backward()
            
            # scaler.unscale_(optimizer)
            # print grad
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"{name}: grad norm = {param.grad.norm().item():.4f}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            loss.backward()
            optimizer.step()
            
            # scaler.step(optimizer)
            # scaler.update()
            
            with torch.no_grad():
                metric.add(loss_ce.item() * images.size(0), loss_dice.item() * images.size(0), images.size(0))
                
        lr_scheduler.step()
        print(f"Epoch {epoch+1}, CE Loss: {metric[0] / metric[2]:.4f}, Dice Loss: {metric[1] / metric[2]:.4f}")
    
    # save model
    # Create a directory with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_dir = config.LOG_DIR / 'NUNet-MoNuSeg' / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), save_dir / "nested_unet.pth")
    torch.save(model.state_dict(), config.LOG_DIR / 'NUNet-MoNuSeg' / "latest.pth")
    
    # simple test
    model.eval()
    with torch.no_grad():
        path = config.DATA_DIR / 'MoNuSeg' / 'patches' / 'images' / 'TCGA-18-5592-01Z-00-DX1_0_0.png'
        image = cv2.imread(str(path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1))  # Convert to (C, H, W)
        image = image_transforms(image_tensor).unsqueeze(0).to(device)  # Add batch dimension
        output = model(image)
        output = output[0] if isinstance(output, tuple) else output  # Use first output if deep supervision
        indices = torch.argmax(output, dim=1)  # Convert to numpy for visualization
        pred_mask = F.one_hot(indices, num_classes=3).squeeze(0) * 255.0  # Convert to (H, W, C) and scale to [0, 255]
        final_mask = pred_mask.cpu().numpy().astype('uint8')
        final_mask = cv2.cvtColor(final_mask, cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving
        cv2.imwrite(str(config.DATA_DIR / 'TCGA-18-5592-01Z-00-DX1_0_0_pred.png'), final_mask)
    