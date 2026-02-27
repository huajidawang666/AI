"""
╔══════════════════════════════════════════════════════════════╗
║          Mask R-CNN From Scratch — PyTorch 实现              ║
║                                                              ║
║  模块结构：                                                    ║
║    1. ResNet Backbone + FPN  (多尺度特征提取)                  ║
║    2. Anchor Generator       (生成先验框)                      ║
║    3. RPN                    (候选区域生成)                    ║
║    4. ROI Align              (精准特征提取，双线性插值)          ║
║    5. Box Head               (分类 + BBox 回归)               ║
║    6. Mask Head              (像素级实例分割)                  ║
║    7. 完整 MaskRCNN 模型                                       ║
║    8. Demo（合成数据验证前向传播）                              ║
╚══════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops import nms
import math
from typing import List, Dict, Optional, Tuple


# ================================================================
# 1. FPN (Feature Pyramid Network)
# ================================================================

class FPN(nn.Module):
    """
    特征金字塔网络
    
    将 ResNet 各阶段输出 (C2~C5) 融合为统一 256 通道的多尺度特征图。
    
    结构：
      C2(256ch) ──┐
      C3(512ch) ──┤  1x1 lateral conv → 统一到 256ch
      C4(1024ch)──┤  + top-down upsample 融合高层语义
      C5(2048ch)──┘  → 3x3 output conv 平滑 → P2, P3, P4, P5
    
    好处：低层特征有精确位置，高层特征有丰富语义，FPN 两者兼得。
    """

    def __init__(self, in_channels_list: List[int], out_channels: int = 256):
        super().__init__()
        # 横向连接：将各层通道数统一为 out_channels
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])
        # 输出卷积：3x3 消除融合时的混叠效应
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """
        输入: {'C2': Tensor, 'C3': Tensor, 'C4': Tensor, 'C5': Tensor}
        输出: [P2, P3, P4, P5]  — 各层均为 256 通道
        """
        feat_list = list(features.values())  # [C2, C3, C4, C5]

        # Step 1: 横向连接，统一通道数
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, feat_list)]

        # Step 2: Top-down 融合（从高层向低层传播语义信息）
        for i in range(len(laterals) - 1, 0, -1):
            # 将高层特征上采样到低层尺寸，再相加
            upsampled = F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[-2:],
                mode='nearest'
            )
            laterals[i - 1] = laterals[i - 1] + upsampled

        # Step 3: 3x3 卷积平滑输出
        outputs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]
        return outputs  # [P2, P3, P4, P5]


class ResNetBackboneWithFPN(nn.Module):
    """
    ResNet50 Backbone + FPN

    ResNet 各阶段的输出尺寸 (输入 512x512 为例)：
      layer0 → 128x128  (stride=4)
      layer1 → 128x128  C2  256ch
      layer2 →  64x64   C3  512ch
      layer3 →  32x32   C4  1024ch
      layer4 →  16x16   C5  2048ch
    """

    def __init__(self, pretrained: bool = False):
        super().__init__()
        resnet = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None
        )

        # 拆分 ResNet 各阶段，方便提取中间特征
        self.layer0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1  # C2
        self.layer2 = resnet.layer2  # C3
        self.layer3 = resnet.layer3  # C4
        self.layer4 = resnet.layer4  # C5

        self.fpn = FPN(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=256
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """输入: (B, 3, H, W) → 输出: [P2, P3, P4, P5]"""
        x  = self.layer0(x)
        c2 = self.layer1(x)   # (B, 256,  H/4,  W/4)
        c3 = self.layer2(c2)  # (B, 512,  H/8,  W/8)
        c4 = self.layer3(c3)  # (B, 1024, H/16, W/16)
        c5 = self.layer4(c4)  # (B, 2048, H/32, W/32)

        fpn_features = self.fpn({'C2': c2, 'C3': c3, 'C4': c4, 'C5': c5})
        return fpn_features  # [P2, P3, P4, P5]，每层 256 通道


# ================================================================
# 2. Anchor Generator
# ================================================================

class AnchorGenerator(nn.Module):
    """
    在 FPN 各层特征图的每个位置生成先验框（Anchors）。

    每层特征图对应不同大小的 anchor：
      P2 (stride=4)  → anchor size: 32²
      P3 (stride=8)  → anchor size: 64²
      P4 (stride=16) → anchor size: 128²
      P5 (stride=32) → anchor size: 256²

    每个位置生成 3 种宽高比 (0.5, 1.0, 2.0)，共 3 个 anchor。
    """

    def __init__(
        self,
        sizes: Tuple = ((32,), (64,), (128,), (256,)),
        aspect_ratios: Tuple = ((0.5, 1.0, 2.0),) * 4,
        strides: Tuple = (4, 8, 16, 32)
    ):
        super().__init__()
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.strides = strides

        # 预计算每层的基础 anchor（中心在原点）
        self.cell_anchors = [
            self._make_base_anchors(sz, ar)
            for sz, ar in zip(sizes, aspect_ratios)
        ]

    @staticmethod
    def _make_base_anchors(sizes, aspect_ratios) -> torch.Tensor:
        """
        生成单个位置上的所有 anchor。

        对于 size=s, ratio=r：
          w = s / sqrt(r)，h = s * sqrt(r)
        这样保证 w*h = s² 面积不变，同时 w/h = 1/r。

        返回: (num_anchors, 4) — [x1, y1, x2, y2]，中心在原点
        """
        sizes = torch.tensor(sizes, dtype=torch.float32)
        ratios = torch.tensor(aspect_ratios, dtype=torch.float32)

        h_ratios = torch.sqrt(ratios)
        w_ratios = 1.0 / h_ratios

        ws = (w_ratios[:, None] * sizes[None, :]).view(-1)
        hs = (h_ratios[:, None] * sizes[None, :]).view(-1)

        base_anchors = torch.stack(
            [-ws / 2, -hs / 2, ws / 2, hs / 2], dim=1
        )  # (A, 4)
        return base_anchors

    def forward(
        self,
        feature_maps: List[torch.Tensor],
        image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        为所有 FPN 层生成 anchor。

        原理：特征图上每个位置 (i, j) 对应原图坐标 (i*stride, j*stride)，
              在该位置放置 A 个不同形状的 anchor。

        输出: (N_total, 4) — 所有层所有位置的 anchor，图像坐标系
        """
        all_anchors = []

        for i, (feat, stride) in enumerate(zip(feature_maps, self.strides)):
            _, _, H, W = feat.shape
            device = feat.device

            # 特征图每个位置对应的原图中心坐标
            shift_x = (torch.arange(W, device=device) + 0.5) * stride
            shift_y = (torch.arange(H, device=device) + 0.5) * stride
            # meshgrid 生成所有位置组合
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
            shifts = torch.stack([
                shift_x.flatten(), shift_y.flatten(),
                shift_x.flatten(), shift_y.flatten()
            ], dim=1)  # (H*W, 4)

            # 基础 anchor + 位置偏移 → 实际 anchor
            cell_anch = self.cell_anchors[i].to(device)  # (A, 4)
            # 广播相加: (H*W, 1, 4) + (1, A, 4) → (H*W, A, 4)
            anchors = (shifts[:, None, :] + cell_anch[None, :, :]).view(-1, 4)
            all_anchors.append(anchors)

        return torch.cat(all_anchors, dim=0)  # (N_total, 4)


# ================================================================
# 3. RPN (Region Proposal Network)
# ================================================================

class RPNHead(nn.Module):
    """
    RPN 预测头：对每个 anchor 预测「是否前景」和「位置偏移」。

    结构：
      3x3 conv → ReLU
        ↓           ↓
      1x1 conv    1x1 conv
    (objectness)  (bbox δ)
    """

    def __init__(self, in_channels: int = 256, num_anchors: int = 3):
        super().__init__()
        # 共享的 3x3 卷积，提取 anchor 局部上下文
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        # objectness: 每个 anchor 一个分数（是否含目标）
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, 1)
        # bbox delta: 每个 anchor 4 个偏移量 (tx, ty, tw, th)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, 1)

        for layer in [self.conv, self.cls_logits, self.bbox_pred]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.zeros_(layer.bias)

    def forward(self, features: List[torch.Tensor]):
        """输出每层的 cls_scores 和 bbox_deltas"""
        cls_scores, bbox_deltas = [], []
        for feat in features:
            t = F.relu(self.conv(feat))
            cls_scores.append(self.cls_logits(t))
            bbox_deltas.append(self.bbox_pred(t))
        return cls_scores, bbox_deltas


class RPN(nn.Module):
    """
    完整 RPN 流程：

    生成 anchor → RPN Head 预测 → 应用 delta → clip → NMS → proposals
    """

    def __init__(
        self,
        anchor_generator: AnchorGenerator,
        rpn_head: RPNHead,
        pre_nms_top_n: int = 2000,
        post_nms_top_n: int = 1000,
        nms_thresh: float = 0.7,
    ):
        super().__init__()
        self.anchor_generator = anchor_generator
        self.head = rpn_head
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh

    @staticmethod
    def apply_deltas(anchors: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        """
        将预测的偏移量 delta 应用到 anchor 上，得到预测框。

        R-CNN 的 bbox parameterization：
          tx = (pred_cx - anchor_cx) / anchor_w
          ty = (pred_cy - anchor_cy) / anchor_h
          tw = log(pred_w / anchor_w)
          th = log(pred_h / anchor_h)

        逆变换（这里做的）：
          pred_cx = tx * anchor_w + anchor_cx
          pred_w  = exp(tw) * anchor_w
        """
        widths  = anchors[:, 2] - anchors[:, 0]
        heights = anchors[:, 3] - anchors[:, 1]
        cx = anchors[:, 0] + 0.5 * widths
        cy = anchors[:, 1] + 0.5 * heights

        dx, dy = deltas[:, 0], deltas[:, 1]
        dw = deltas[:, 2].clamp(max=math.log(1000.0 / 16))  # 防止 exp 溢出
        dh = deltas[:, 3].clamp(max=math.log(1000.0 / 16))

        pred_cx = dx * widths  + cx
        pred_cy = dy * heights + cy
        pred_w  = torch.exp(dw) * widths
        pred_h  = torch.exp(dh) * heights

        return torch.stack([
            pred_cx - 0.5 * pred_w,
            pred_cy - 0.5 * pred_h,
            pred_cx + 0.5 * pred_w,
            pred_cy + 0.5 * pred_h,
        ], dim=1)

    @staticmethod
    def clip_boxes(boxes: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        """裁剪越界框到图像边界内"""
        H, W = image_size
        return torch.stack([
            boxes[:, 0].clamp(0, W),
            boxes[:, 1].clamp(0, H),
            boxes[:, 2].clamp(0, W),
            boxes[:, 3].clamp(0, H),
        ], dim=1)

    def forward(
        self,
        features: List[torch.Tensor],
        image_size: Tuple[int, int],
        targets=None
    ) -> Tuple[torch.Tensor, dict]:
        """
        输入: FPN 特征图列表，图像尺寸
        输出: proposals (N, 4)，rpn_losses dict（训练时才有 loss）
        """
        # 1. RPN Head 预测各层的 objectness 和 delta
        cls_scores, bbox_deltas = self.head(features)

        # 2. 生成所有 anchor
        anchors = self.anchor_generator(features, image_size)  # (N_total, 4)

        # 3. 将各层预测展平并拼接（取 batch=0）
        all_scores = torch.cat([
            s.permute(0, 2, 3, 1).reshape(s.shape[0], -1)  # (B, H*W*A)
            for s in cls_scores
        ], dim=1)[0]  # (N_total,)

        all_deltas = torch.cat([
            d.permute(0, 2, 3, 1).reshape(d.shape[0], -1, 4)  # (B, H*W*A, 4)
            for d in bbox_deltas
        ], dim=1)[0]  # (N_total, 4)

        # 4. 选 top-K 个高分 anchor，避免对所有 anchor 做 NMS
        scores = torch.sigmoid(all_scores)
        topk_n = min(self.pre_nms_top_n, scores.shape[0])
        _, topk_idx = scores.topk(topk_n)

        topk_scores  = scores[topk_idx]
        topk_anchors = anchors[topk_idx]
        topk_deltas  = all_deltas[topk_idx]

        # 5. 应用 delta，得到预测框
        proposals = self.apply_deltas(topk_anchors, topk_deltas)
        proposals = self.clip_boxes(proposals, image_size)

        # 6. NMS：去掉重叠度高的框，保留最多 post_nms_top_n 个
        keep = nms(proposals, topk_scores, self.nms_thresh)
        keep = keep[:self.post_nms_top_n]
        proposals = proposals[keep]

        return proposals, {}  # {} 占位，完整训练需在此返回 RPN loss


# ================================================================
# 4. ROI Align（双线性插值，从零实现）
# ================================================================

class ROIAlign(nn.Module):
    """
    ROI Align：无精度损失地从特征图上提取每个 ROI 的特征。

    ROI Pooling 的问题：
      - 两次量化取整（ROI 坐标 → 特征坐标，bin 边界取整）
      - 对 BBox 检测影响不大，但对像素级 Mask 误差很大

    ROI Align 的改进：
      - 不做任何量化，直接用浮点坐标
      - 每个 bin 内采样若干点，用双线性插值取值，再平均

    实现方式：利用 F.grid_sample（双线性插值）实现等价效果
    """

    def __init__(
        self,
        output_size: int = 7,
        spatial_scale: float = 1 / 4,
        sampling_ratio: int = 2
    ):
        super().__init__()
        self.output_size = output_size      # 输出特征图大小 (7×7 or 14×14)
        self.spatial_scale = spatial_scale  # 特征图相对原图的缩放比 (1/4 for P2)
        self.sampling_ratio = sampling_ratio

    def forward(
        self,
        feature_map: torch.Tensor,  # (1, C, H, W)
        rois: torch.Tensor          # (N, 4) 原图坐标
    ) -> torch.Tensor:
        """
        输出: (N, C, output_size, output_size)
        """
        # 将 ROI 坐标缩放到特征图坐标系
        rois_feat = rois * self.spatial_scale  # (N, 4)

        N = rois_feat.shape[0]
        _, C, H, W = feature_map.shape
        output_size = self.output_size

        results = []
        for i in range(N):
            x1, y1, x2, y2 = rois_feat[i]
            roi_w = (x2 - x1).clamp(min=1e-3)
            roi_h = (y2 - y1).clamp(min=1e-3)

            # 在 ROI 内均匀采样 output_size 个点（bin 中心）
            # 每个 bin 的中心点坐标（特征图坐标系）
            xs = torch.linspace(
                float(x1 + roi_w / (2 * output_size)),
                float(x2 - roi_w / (2 * output_size)),
                output_size,
                device=feature_map.device
            )
            ys = torch.linspace(
                float(y1 + roi_h / (2 * output_size)),
                float(y2 - roi_h / (2 * output_size)),
                output_size,
                device=feature_map.device
            )

            # F.grid_sample 要求归一化坐标 [-1, 1]
            # x_norm = (x / (W-1)) * 2 - 1
            xs_norm = (xs / (W - 1)) * 2 - 1
            ys_norm = (ys / (H - 1)) * 2 - 1

            grid_y, grid_x = torch.meshgrid(ys_norm, xs_norm, indexing='ij')
            grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, oh, ow, 2)

            # 双线性插值采样
            sampled = F.grid_sample(
                feature_map,       # (1, C, H, W)
                grid,              # (1, oh, ow, 2)
                mode='bilinear',
                align_corners=True,
                padding_mode='zeros'
            )  # (1, C, oh, ow)
            results.append(sampled[0])  # (C, oh, ow)

        return torch.stack(results, dim=0)  # (N, C, oh, ow)


# ================================================================
# 5. Box Head（分类 + BBox 回归）
# ================================================================

class BoxHead(nn.Module):
    """
    对每个 ROI 预测：
      - 类别得分 (num_classes 维)
      - 精细的 BBox 偏移量 (num_classes * 4 维，每类独立回归)

    输入：ROI Align 提取的 7×7 特征图
    结构：Flatten → FC1024 → FC1024 → 分类/回归分支
    """

    def __init__(self, in_channels: int = 256, num_classes: int = 91):
        super().__init__()
        self.fc6 = nn.Linear(in_channels * 7 * 7, 1024)
        self.fc7 = nn.Linear(1024, 1024)

        self.cls_score  = nn.Linear(1024, num_classes)       # 分类打分
        self.bbox_pred  = nn.Linear(1024, num_classes * 4)   # 每类独立回归

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.zeros_(self.cls_score.bias)
        nn.init.zeros_(self.bbox_pred.bias)

    def forward(self, x: torch.Tensor):
        """
        x: (N, 256, 7, 7)
        return: cls_logits (N, C), bbox_deltas (N, C*4)
        """
        x = x.flatten(1)           # (N, 256*7*7)
        x = F.relu(self.fc6(x))    # (N, 1024)
        x = F.relu(self.fc7(x))    # (N, 1024)
        return self.cls_score(x), self.bbox_pred(x)


# ================================================================
# 6. Mask Head（像素级实例分割）
# ================================================================

class MaskHead(nn.Module):
    """
    对每个 ROI 预测像素级二值 Mask。

    关键设计：
      - 输入用 14×14（比 Box Head 的 7×7 更大），保留更多空间信息
      - 4 层 3x3 卷积保持分辨率，提取空间特征
      - 转置卷积（反卷积）上采样到 28×28
      - 每个类别独立预测一个 Mask，互不干扰

    Loss 计算（代码略）：
      只对 GT 类别对应的那个 Mask 计算 Binary Cross Entropy，
      其他类别的 Mask 不参与反向传播。
      这是 Mask 分支与分类解耦的关键！
    """

    def __init__(self, in_channels: int = 256, num_classes: int = 91):
        super().__init__()

        # 4 层 3x3 卷积，保持 14×14 不变
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(in_channels, in_channels, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        self.convs = nn.Sequential(*layers)

        # 反卷积：14×14 → 28×28（步长=2，核=2）
        self.deconv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)

        # 每类输出一个通道的 Mask（不经过 sigmoid，loss 中用 BCE with logits）
        self.mask_pred = nn.Conv2d(in_channels, num_classes, kernel_size=1)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, 256, 14, 14)
        return: (N, num_classes, 28, 28) — 每类一个 Mask logits
        """
        x = self.convs(x)           # (N, 256, 14, 14)
        x = F.relu(self.deconv(x))  # (N, 256, 28, 28)
        x = self.mask_pred(x)       # (N, num_classes, 28, 28)
        return x


# ================================================================
# 7. 完整 Mask R-CNN
# ================================================================

class MaskRCNN(nn.Module):
    """
    完整 Mask R-CNN

    前向流程：
    ┌─────────────────────────────────────────────────────────┐
    │ Image (B,3,H,W)                                         │
    │    ↓                                                    │
    │ [Backbone + FPN] → P2, P3, P4, P5 (多尺度特征)         │
    │    ↓                                                    │
    │ [RPN] → proposals (候选框, 约1000个)                    │
    │    ↓                ↓                                   │
    │ [ROI Align 7x7]  [ROI Align 14x14]                     │
    │    ↓                  ↓                                 │
    │ [Box Head]        [Mask Head]                           │
    │  ├─ cls_logits     → (N, num_classes, 28, 28) Mask      │
    │  └─ bbox_deltas                                         │
    └─────────────────────────────────────────────────────────┘

    损失函数（训练时）：
      L_total = L_rpn_cls + L_rpn_box + L_cls + L_box + L_mask
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # ── Backbone ──────────────────────────────────
        self.backbone = ResNetBackboneWithFPN(pretrained=False)

        # ── RPN ───────────────────────────────────────
        anchor_gen = AnchorGenerator(
            sizes=((32,), (64,), (128,), (256,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 4
        )
        rpn_head = RPNHead(in_channels=256, num_anchors=3)
        self.rpn = RPN(anchor_gen, rpn_head)

        # ── ROI Align ─────────────────────────────────
        # Box Head 用 7×7，Mask Head 用 14×14（更精细）
        self.roi_align_box  = ROIAlign(output_size=7,  spatial_scale=1 / 4)
        self.roi_align_mask = ROIAlign(output_size=14, spatial_scale=1 / 4)

        # ── Heads ─────────────────────────────────────
        self.box_head  = BoxHead(in_channels=256, num_classes=num_classes)
        self.mask_head = MaskHead(in_channels=256, num_classes=num_classes)

    def forward(
        self,
        images: torch.Tensor,
        targets=None
    ) -> dict:
        """
        images:  (B, 3, H, W)  — 目前仅支持 B=1（demo 模式）
        targets: 训练时传入 GT，推理时为 None

        返回字典包含：
          proposals   — RPN 生成的候选框 (N, 4)
          cls_logits  — 分类得分 (N, num_classes)
          bbox_deltas — 框回归偏移 (N, num_classes*4)
          mask_logits — Mask 预测 (N, num_classes, 28, 28)
        """
        assert images.dim() == 4, "期望输入形状: (B, C, H, W)"
        image_size = (images.shape[2], images.shape[3])  # (H, W)

        # ① Backbone + FPN
        features = self.backbone(images)  # [P2, P3, P4, P5]

        # ② RPN → proposals
        proposals, rpn_losses = self.rpn(features, image_size, targets)

        if proposals.shape[0] == 0:
            print("  [警告] RPN 未生成任何 proposal")
            return {'proposals': proposals,
                    'cls_logits': None,
                    'bbox_deltas': None,
                    'mask_logits': None}

        # ③ ROI Align：从 P2 提取特征（简化，完整实现需按框大小选层）
        #    完整实现：根据 ROI 面积分配到对应 FPN 层级（FPN paper 公式）
        p2 = features[0]  # P2: stride=4，分辨率最高，适合小目标
        roi_feat_box  = self.roi_align_box(p2, proposals)   # (N, 256, 7,  7)
        roi_feat_mask = self.roi_align_mask(p2, proposals)  # (N, 256, 14, 14)

        # ④ Box Head → 分类 + 回归
        cls_logits, bbox_deltas = self.box_head(roi_feat_box)

        # ⑤ Mask Head → 像素级分割
        mask_logits = self.mask_head(roi_feat_mask)

        return {
            'proposals':   proposals,    # (N, 4)
            'cls_logits':  cls_logits,   # (N, num_classes)
            'bbox_deltas': bbox_deltas,  # (N, num_classes*4)
            'mask_logits': mask_logits,  # (N, num_classes, 28, 28)
        }


# ================================================================
# 8. Demo：合成数据验证前向传播
# ================================================================

def count_params(model: nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"总参数: {total/1e6:.1f}M  |  可训练: {trainable/1e6:.1f}M"


def demo():
    SEP = "=" * 62
    print(SEP)
    print("   Mask R-CNN From Scratch — PyTorch Demo")
    print(SEP)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'设备':>10}: {device}")

    # ── 构建模型 ─────────────────────────────────────────────────
    print(f"\n{'步骤 1':>10}: 构建模型 (num_classes=10)")
    model = MaskRCNN(num_classes=10).to(device)
    model.eval()
    print(f"{'参数量':>10}: {count_params(model)}")

    # ── 模型结构 ─────────────────────────────────────────────────
    print(f"\n{'步骤 2':>10}: 模型子模块")
    for name, module in model.named_children():
        n_params = sum(p.numel() for p in module.parameters()) / 1e6
        print(f"  ├─ {name:<20} {n_params:.1f}M params")

    # ── 合成输入 ─────────────────────────────────────────────────
    print(f"\n{'步骤 3':>10}: 生成合成图像 (1, 3, 512, 512)")
    dummy_image = torch.randn(1, 3, 512, 512, device=device)
    print(f"{'输入范围':>10}: min={dummy_image.min():.2f}, max={dummy_image.max():.2f}")

    # ── 前向传播 ─────────────────────────────────────────────────
    print(f"\n{'步骤 4':>10}: 前向传播中...")
    with torch.no_grad():
        output = model(dummy_image)
    print(f"{'状态':>10}: ✅ 成功！")

    # ── 输出分析 ─────────────────────────────────────────────────
    print(f"\n{'步骤 5':>10}: 各阶段输出形状")
    props = output['proposals']
    print(f"  ├─ RPN proposals   : {tuple(props.shape)}  ({props.shape[0]} 个候选框)")

    if output['cls_logits'] is not None:
        cls  = output['cls_logits']
        bbox = output['bbox_deltas']
        mask = output['mask_logits']
        print(f"  ├─ 分类 logits     : {tuple(cls.shape)}")
        print(f"  ├─ BBox deltas     : {tuple(bbox.shape)}")
        print(f"  └─ Mask logits     : {tuple(mask.shape)}")

        # ── 简单推理 ─────────────────────────────────────────────
        print(f"\n{'步骤 6':>10}: 简单推理（取置信度最高的框）")
        scores = cls.softmax(dim=-1)
        top_score, top_cls = scores.max(dim=-1)
        best_idx = top_score.argmax().item()

        box = props[best_idx].cpu().numpy().round(1)
        print(f"  ├─ 最高分 idx      : {best_idx}")
        print(f"  ├─ 预测框 [x1,y1,x2,y2]: {box}")
        print(f"  ├─ 预测类别        : {top_cls[best_idx].item()}")
        print(f"  ├─ 置信度          : {top_score[best_idx].item():.4f}")
        print(f"  └─ 对应 Mask 形状  : {tuple(mask[best_idx].shape)}")

    # ── 下一步指引 ───────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  ✅ 模型前向传播验证完毕！")
    print(SEP)
    print("\n📌 下一步（接入真实数据）:")
    print("  1. 数据集：下载 COCO 或准备自定义数据，用 COCO API 加载标注")
    print("  2. Loss：实现 RPN loss (BCE + SmoothL1)")
    print("           BoxHead loss (CrossEntropy + SmoothL1)")
    print("           MaskHead loss (BCE with logits，只对 GT 类别计算)")
    print("  3. 训练：SGD + Warmup LR，先冻结 backbone 再 fine-tune")
    print("  4. 后处理：BoxHead 输出 → 应用 delta → NMS → 最终检测框")
    print("             Mask logits → sigmoid → 阈值 0.5 → 二值 Mask")
    print()


if __name__ == '__main__':
    demo()