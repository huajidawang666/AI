"""
DeepLabv3 Implementation in PyTorch
论文: Rethinking Atrous Convolution for Semantic Image Segmentation (Chen et al., 2017)

主要组件:
  - Atrous/Dilated Convolution（空洞卷积）
  - ASPP (Atrous Spatial Pyramid Pooling)
  - ResNet-50/101 Backbone
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet101, ResNet50_Weights, ResNet101_Weights


# ─────────────────────────────────────────────
# 1. ASPP 模块
# ─────────────────────────────────────────────
class ASPPConv(nn.Sequential):
    """单个空洞卷积分支"""
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3,
                      padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ASPPPooling(nn.Module):
    """全局平均池化分支"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        x = self.pool(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling
    汇聚多尺度上下文信息
    默认 dilations: [6, 12, 18]（输入 stride=16 时）
    """
    def __init__(self, in_channels: int, out_channels: int = 256,
                 dilations=(6, 12, 18)):
        super().__init__()

        # 1×1 conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # 空洞卷积分支
        self.aspp_convs = nn.ModuleList([
            ASPPConv(in_channels, out_channels, d) for d in dilations
        ])

        # 全局池化分支
        self.global_pool = ASPPPooling(in_channels, out_channels)

        # 融合：(1 + len(dilations) + 1) 个分支
        num_branches = 1 + len(dilations) + 1
        self.project = nn.Sequential(
            nn.Conv2d(num_branches * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [self.conv1(x)]
        features += [conv(x) for conv in self.aspp_convs]
        features.append(self.global_pool(x))
        return self.project(torch.cat(features, dim=1))


# ─────────────────────────────────────────────
# 2. DeepLabv3 主体
# ─────────────────────────────────────────────
class DeepLabV3(nn.Module):
    """
    DeepLabv3 语义分割模型

    Args:
        num_classes: 分割类别数（含背景）
        backbone:    'resnet50' | 'resnet101'
        output_stride: 8 或 16（控制空洞卷积率）
        pretrained:  是否使用 ImageNet 预训练权重
    """

    def __init__(self, num_classes: int = 21,
                 backbone: str = "resnet50",
                 output_stride: int = 16,
                 pretrained: bool = True):
        super().__init__()

        assert output_stride in (8, 16), "output_stride 必须为 8 或 16"
        self.output_stride = output_stride

        # ── Backbone ──────────────────────────────
        if backbone == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            base = resnet50(weights=weights)
            aspp_in = 2048
        elif backbone == "resnet101":
            weights = ResNet101_Weights.DEFAULT if pretrained else None
            base = resnet101(weights=weights)
            aspp_in = 2048
        else:
            raise ValueError(f"不支持的 backbone: {backbone}")

        # 修改 layer3/layer4 以使用空洞卷积
        # output_stride=16 → layer4 dilation=2
        # output_stride=8  → layer3 dilation=2, layer4 dilation=4
        if output_stride == 16:
            self._set_dilations(base.layer3, stride=1, dilation=1)
            self._set_dilations(base.layer4, stride=1, dilation=2)
        else:  # output_stride == 8
            self._set_dilations(base.layer3, stride=1, dilation=2)
            self._set_dilations(base.layer4, stride=1, dilation=4)

        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
        )

        # ── ASPP ──────────────────────────────────
        if output_stride == 16:
            dilations = (6, 12, 18)
        else:  # output_stride == 8，特征图更大，需要更大 dilation
            dilations = (12, 24, 36)

        self.aspp = ASPP(aspp_in, 256, dilations)

        # ── 分类头 ────────────────────────────────
        self.classifier = nn.Conv2d(256, num_classes, 1)

        self._init_weights()

    # ── 工具函数 ──────────────────────────────────
    @staticmethod
    def _set_dilations(layer: nn.Module, stride: int, dilation: int):
        """将 ResNet stage 中的 stride 替换为空洞卷积"""
        for name, m in layer.named_modules():
            if isinstance(m, nn.Conv2d):
                if m.stride == (2, 2):
                    m.stride = (stride, stride)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilation, dilation)
                    m.padding = (dilation, dilation)

    def _init_weights(self):
        for m in [self.aspp, self.classifier]:
            for mod in m.modules():
                if isinstance(mod, nn.Conv2d):
                    nn.init.kaiming_normal_(mod.weight, mode="fan_out",
                                           nonlinearity="relu")
                elif isinstance(mod, nn.BatchNorm2d):
                    nn.init.constant_(mod.weight, 1)
                    nn.init.constant_(mod.bias, 0)

    # ── 前向传播 ──────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]        # H × W

        x = self.backbone(x)             # (B, 2048, H/OS, W/OS)
        x = self.aspp(x)                 # (B, 256,  H/OS, W/OS)
        x = self.classifier(x)           # (B, C,    H/OS, W/OS)

        # 双线性上采样回原始分辨率
        x = F.interpolate(x, size=input_size,
                          mode="bilinear", align_corners=False)
        return x                         # (B, num_classes, H, W)


# ─────────────────────────────────────────────
# 3. 工厂函数
# ─────────────────────────────────────────────
def deeplabv3_resnet50(num_classes=21, output_stride=16, pretrained=True):
    return DeepLabV3(num_classes, "resnet50", output_stride, pretrained)


def deeplabv3_resnet101(num_classes=21, output_stride=16, pretrained=True):
    return DeepLabV3(num_classes, "resnet101", output_stride, pretrained)


# ─────────────────────────────────────────────
# 4. 训练工具
# ─────────────────────────────────────────────
class SegmentationLoss(nn.Module):
    """交叉熵 + 可选 Dice Loss"""
    def __init__(self, ignore_index=255, use_dice=False):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.use_dice = use_dice

    def dice_loss(self, pred, target, num_classes, ignore_index=255, eps=1e-6):
        pred = F.softmax(pred, dim=1)
        loss = 0.0
        for c in range(num_classes):
            mask = (target != ignore_index)
            tc = ((target == c) & mask).float()
            pc = pred[:, c][mask]
            tc = tc[mask]
            inter = (pc * tc).sum()
            loss += 1 - (2 * inter + eps) / (pc.sum() + tc.sum() + eps)
        return loss / num_classes

    def forward(self, pred, target):
        loss = self.ce(pred, target)
        if self.use_dice:
            loss += self.dice_loss(pred, target, pred.shape[1])
        return loss


def build_optimizer(model: DeepLabV3, lr=0.01, weight_decay=1e-4):
    """主干网络使用 1/10 学习率"""
    backbone_params = list(model.backbone.parameters())
    head_params = (list(model.aspp.parameters()) +
                   list(model.classifier.parameters()))
    return torch.optim.SGD(
        [{"params": backbone_params, "lr": lr * 0.1},
         {"params": head_params,    "lr": lr}],
        momentum=0.9, weight_decay=weight_decay,
    )


def poly_lr_scheduler(optimizer, base_lr, step, max_steps, power=0.9):
    """Poly 学习率衰减策略"""
    factor = (1 - step / max_steps) ** power
    for i, pg in enumerate(optimizer.param_groups):
        pg["lr"] = base_lr * (0.1 if i == 0 else 1.0) * factor


# ─────────────────────────────────────────────
# 5. 快速验证
# ─────────────────────────────────────────────
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    model = deeplabv3_resnet50(num_classes=21, output_stride=16, pretrained=False)
    model = model.to(device).eval()

    x = torch.randn(2, 3, 512, 512).to(device)
    with torch.no_grad():
        out = model(x)

    print(f"输入形状:  {tuple(x.shape)}")
    print(f"输出形状:  {tuple(out.shape)}")  # 应为 (2, 21, 512, 512)

    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"参数量:    {total:.1f}M")