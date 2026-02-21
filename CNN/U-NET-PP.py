"""
U-Net++ (Nested U-Net) implementation in PyTorch.

Key features:
- Nested skip connections with dense semantic aggregation.
- Optional deep supervision (returns multi-scale outputs).
- Configurable encoder width and normalization.

Reference:
Zhou et al., "UNet++: A Nested U-Net Architecture for Medical Image Segmentation" (2018).
"""

from typing import List, Tuple, Union

from torchvision.transforms import v2
from dataset.MoNuSeg import MoNuSegDataset
import torch
from torch import nn
import torch.nn.functional as F

NUM_EPOCHS = 10

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

    def _up(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=target.shape[2:], mode="bilinear", align_corners=False)

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

def get_dataloader(batch_size: int = 64):
    train_transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])
    
    dataset = MoNuSegDataset(transform=train_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

dataloader = get_dataloader()
# quick test
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = NestedUNet(in_channels=3, num_classes=3, deep_supervision=True)
    model.to(device)
    
    weights = torch.tensor([1.0, 1.0, 2.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            targets = torch.argmax(masks, dim=1)  # Assuming masks are one-hot encoded
            outputs = model(images)
            
            if isinstance(outputs, tuple):
                loss = sum(criterion(out, targets) for out in outputs) / len(outputs)
            else:
                loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    