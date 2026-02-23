"""
U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection
Paper: https://arxiv.org/abs/2005.09007

Architecture:
  - Encoder: 6 stages (En_1 ~ En_6)
  - Decoder: 6 stages (De_1 ~ De_6) with supervision heads
  - Each stage uses RSU (Recurrent Residual U-block) or RSU4F (dilated version)
  - Final output is a fused saliency map from all 6 side outputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
#  Basic building block
# ─────────────────────────────────────────────

class ConvBNReLU(nn.Module):
    """Conv2d + BN + ReLU, with optional dilation."""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size,
                      padding=padding * dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ─────────────────────────────────────────────
#  RSU-L  (Recurrent Residual U-block, depth L)
# ─────────────────────────────────────────────

class RSU(nn.Module):
    """
    RSU-L block with L encoder stages and symmetric decoder.
    Args:
        height (int): number of encoder/decoder levels (L)
        in_ch  (int): input channels
        mid_ch (int): intermediate channels inside the U
        out_ch (int): output channels of the whole block
    """
    def __init__(self, height: int, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        assert height >= 2

        self.rebnconvin = ConvBNReLU(in_ch, out_ch)

        # Encoder
        self.encoders = nn.ModuleList()
        self.pools    = nn.ModuleList()
        for i in range(height):
            if i == 0:
                self.encoders.append(ConvBNReLU(out_ch, mid_ch))
            else:
                self.encoders.append(ConvBNReLU(mid_ch, mid_ch))
            if i < height - 1:
                self.pools.append(nn.MaxPool2d(2, stride=2, ceil_mode=True))

        # Bottleneck (deepest level)
        self.bottleneck = ConvBNReLU(mid_ch, mid_ch, dilation=2, padding=2)

        # Decoder
        self.decoders = nn.ModuleList()
        for i in range(height - 1):
            self.decoders.append(ConvBNReLU(mid_ch * 2, mid_ch))

        self.rebnconvout = ConvBNReLU(mid_ch * 2, out_ch)

    def forward(self, x):
        hx = self.rebnconvin(x)

        # Encode
        enc_feats = []
        h = hx
        for i, enc in enumerate(self.encoders):
            h = enc(h)
            enc_feats.append(h)
            if i < len(self.pools):
                h = self.pools[i](h)

        # Bottleneck
        h = self.bottleneck(h)

        # Decode (reverse skip connections)
        for i, dec in enumerate(self.decoders):
            skip = enc_feats[-(i + 1)]
            h = F.interpolate(h, size=skip.shape[2:], mode='bilinear', align_corners=False)
            h = dec(torch.cat([h, skip], dim=1))

        # Final concat with top-level encoder feature
        h = F.interpolate(h, size=enc_feats[0].shape[2:], mode='bilinear', align_corners=False)
        h = self.rebnconvout(torch.cat([h, enc_feats[0]], dim=1))

        return h + hx  # residual connection


class RSU4F(nn.Module):
    """
    RSU-4F: dilated version used at the deepest stages (no spatial pooling).
    Dilation rates: 1, 2, 4, 8
    """
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.rebnconvin  = ConvBNReLU(in_ch,  out_ch)
        self.rebnconv1   = ConvBNReLU(out_ch, mid_ch, dilation=1,  padding=1)
        self.rebnconv2   = ConvBNReLU(mid_ch, mid_ch, dilation=2,  padding=2)
        self.rebnconv3   = ConvBNReLU(mid_ch, mid_ch, dilation=4,  padding=4)
        self.rebnconv4   = ConvBNReLU(mid_ch, mid_ch, dilation=8,  padding=8)
        self.rebnconv3d  = ConvBNReLU(mid_ch * 2, mid_ch, dilation=4, padding=4)
        self.rebnconv2d  = ConvBNReLU(mid_ch * 2, mid_ch, dilation=2, padding=2)
        self.rebnconv1d  = ConvBNReLU(mid_ch * 2, out_ch, dilation=1, padding=1)

    def forward(self, x):
        hx  = self.rebnconvin(x)
        hx1 = self.rebnconv1(hx)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.rebnconv4(hx3)
        hx3d = self.rebnconv3d(torch.cat([hx4, hx3], dim=1))
        hx2d = self.rebnconv2d(torch.cat([hx3d, hx2], dim=1))
        hx1d = self.rebnconv1d(torch.cat([hx2d, hx1], dim=1))
        return hx1d + hx


# ─────────────────────────────────────────────
#  Side output head
# ─────────────────────────────────────────────

class SideHead(nn.Module):
    """1×1 conv to produce a single-channel saliency map."""
    def __init__(self, in_ch: int, out_ch: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x, target_size):
        x = self.conv(x)
        return F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)


# ─────────────────────────────────────────────
#  U²-Net (full)
# ─────────────────────────────────────────────

class U2Net(nn.Module):
    """
    U²-Net for salient object detection.
    Input : (B, 3, H, W)
    Output: tuple of 7 sigmoid saliency maps, each (B, 1, H, W)
            (d0 is the fused final prediction; d1~d6 are side outputs for deep supervision)
    """
    def __init__(self, in_ch: int = 3, out_ch: int = 1):
        super().__init__()

        # ── Encoder ──────────────────────────────────
        self.stage1 = RSU(7, in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU(6, 64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU(5, 128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU(4, 256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 256, 512)   # bridge / bottleneck

        # ── Decoder ──────────────────────────────────
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU(4,  1024, 128, 256)
        self.stage3d = RSU(5,   512,  64, 128)
        self.stage2d = RSU(6,   256,  32,  64)
        self.stage1d = RSU(7,   128,  16,  64)

        # ── Side outputs ─────────────────────────────
        self.side1 = SideHead(64,  out_ch)
        self.side2 = SideHead(64,  out_ch)
        self.side3 = SideHead(128, out_ch)
        self.side4 = SideHead(256, out_ch)
        self.side5 = SideHead(512, out_ch)
        self.side6 = SideHead(512, out_ch)

        # ── Fusion conv ──────────────────────────────
        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]

        # ---- Encoder ----
        hx1 = self.stage1(x)
        hx2 = self.stage2(self.pool12(hx1))
        hx3 = self.stage3(self.pool23(hx2))
        hx4 = self.stage4(self.pool34(hx3))
        hx5 = self.stage5(self.pool45(hx4))
        hx6 = self.stage6(self.pool56(hx5))

        # ---- Decoder ----
        hx6up = F.interpolate(hx6, size=hx5.shape[2:], mode='bilinear', align_corners=False)
        hx5d  = self.stage5d(torch.cat([hx6up, hx5], dim=1))

        hx5up = F.interpolate(hx5d, size=hx4.shape[2:], mode='bilinear', align_corners=False)
        hx4d  = self.stage4d(torch.cat([hx5up, hx4], dim=1))

        hx4up = F.interpolate(hx4d, size=hx3.shape[2:], mode='bilinear', align_corners=False)
        hx3d  = self.stage3d(torch.cat([hx4up, hx3], dim=1))

        hx3up = F.interpolate(hx3d, size=hx2.shape[2:], mode='bilinear', align_corners=False)
        hx2d  = self.stage2d(torch.cat([hx3up, hx2], dim=1))

        hx2up = F.interpolate(hx2d, size=hx1.shape[2:], mode='bilinear', align_corners=False)
        hx1d  = self.stage1d(torch.cat([hx2up, hx1], dim=1))

        # ---- Side outputs ----
        d1 = self.side1(hx1d, (H, W))
        d2 = self.side2(hx2d, (H, W))
        d3 = self.side3(hx3d, (H, W))
        d4 = self.side4(hx4d, (H, W))
        d5 = self.side5(hx5d, (H, W))
        d6 = self.side6(hx6,  (H, W))

        # ---- Fusion ----
        d0 = self.outconv(torch.cat([d1, d2, d3, d4, d5, d6], dim=1))

        return (
            torch.sigmoid(d0),
            torch.sigmoid(d1),
            torch.sigmoid(d2),
            torch.sigmoid(d3),
            torch.sigmoid(d4),
            torch.sigmoid(d5),
            torch.sigmoid(d6),
        )


# ─────────────────────────────────────────────
#  U²-Net lite (smaller version)
# ─────────────────────────────────────────────

class U2NetLite(nn.Module):
    """
    U²-Net-lite: lighter variant (~4.7M params vs ~44M for full version).
    Same topology but much smaller channel widths.
    """
    def __init__(self, in_ch: int = 3, out_ch: int = 1):
        super().__init__()

        self.stage1 = RSU(7, in_ch, 16, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU(6, 64, 16, 64)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU(5, 64, 16, 64)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU(4, 64, 16, 64)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(64, 16, 64)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(64, 16, 64)

        self.stage5d = RSU4F(128, 16, 64)
        self.stage4d = RSU(4, 128, 16, 64)
        self.stage3d = RSU(5, 128, 16, 64)
        self.stage2d = RSU(6, 128, 16, 64)
        self.stage1d = RSU(7, 128, 16, 64)

        self.side1 = SideHead(64, out_ch)
        self.side2 = SideHead(64, out_ch)
        self.side3 = SideHead(64, out_ch)
        self.side4 = SideHead(64, out_ch)
        self.side5 = SideHead(64, out_ch)
        self.side6 = SideHead(64, out_ch)

        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]

        hx1 = self.stage1(x)
        hx2 = self.stage2(self.pool12(hx1))
        hx3 = self.stage3(self.pool23(hx2))
        hx4 = self.stage4(self.pool34(hx3))
        hx5 = self.stage5(self.pool45(hx4))
        hx6 = self.stage6(self.pool56(hx5))

        hx6up = F.interpolate(hx6, size=hx5.shape[2:], mode='bilinear', align_corners=False)
        hx5d  = self.stage5d(torch.cat([hx6up, hx5], dim=1))

        hx5up = F.interpolate(hx5d, size=hx4.shape[2:], mode='bilinear', align_corners=False)
        hx4d  = self.stage4d(torch.cat([hx5up, hx4], dim=1))

        hx4up = F.interpolate(hx4d, size=hx3.shape[2:], mode='bilinear', align_corners=False)
        hx3d  = self.stage3d(torch.cat([hx4up, hx3], dim=1))

        hx3up = F.interpolate(hx3d, size=hx2.shape[2:], mode='bilinear', align_corners=False)
        hx2d  = self.stage2d(torch.cat([hx3up, hx2], dim=1))

        hx2up = F.interpolate(hx2d, size=hx1.shape[2:], mode='bilinear', align_corners=False)
        hx1d  = self.stage1d(torch.cat([hx2up, hx1], dim=1))

        d1 = self.side1(hx1d, (H, W))
        d2 = self.side2(hx2d, (H, W))
        d3 = self.side3(hx3d, (H, W))
        d4 = self.side4(hx4d, (H, W))
        d5 = self.side5(hx5d, (H, W))
        d6 = self.side6(hx6,  (H, W))

        d0 = self.outconv(torch.cat([d1, d2, d3, d4, d5, d6], dim=1))

        return (
            torch.sigmoid(d0),
            torch.sigmoid(d1),
            torch.sigmoid(d2),
            torch.sigmoid(d3),
            torch.sigmoid(d4),
            torch.sigmoid(d5),
            torch.sigmoid(d6),
        )


# ─────────────────────────────────────────────
#  Loss function (multi-scale BCE)
# ─────────────────────────────────────────────

def bce_loss(pred, target):
    return F.binary_cross_entropy(pred, target, reduction='mean')


def u2net_loss(preds, target):
    """
    Deep supervision loss: sum of BCE on all 7 outputs.
    preds : tuple of 7 tensors (d0..d6), each (B,1,H,W) after sigmoid
    target: (B, 1, H, W) ground-truth mask in [0,1]
    """
    return sum(bce_loss(p, target) for p in preds)


# ─────────────────────────────────────────────
#  Quick sanity check
# ─────────────────────────────────────────────

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ---- Full model ----
    model = U2Net().to(device)
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"U²-Net       params: {total:.2f} M")

    x = torch.randn(2, 3, 320, 320, device=device)
    with torch.no_grad():
        preds = model(x)
    print(f"Output shape: {preds[0].shape}  (d0)")
    print(f"All outputs : {[p.shape for p in preds]}")

    # ---- Lite model ----
    model_lite = U2NetLite().to(device)
    total_lite = sum(p.numel() for p in model_lite.parameters()) / 1e6
    print(f"\nU²-Net-lite  params: {total_lite:.2f} M")

    with torch.no_grad():
        preds_lite = model_lite(x)
    print(f"Output shape: {preds_lite[0].shape}  (d0)")

    # ---- Loss test ----
    gt = torch.randint(0, 2, (2, 1, 320, 320), dtype=torch.float32, device=device)
    loss = u2net_loss(preds, gt)
    print(f"\nLoss: {loss.item():.4f}")