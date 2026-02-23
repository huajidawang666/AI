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
from torchvision import tv_tensors

NUM_EPOCHS = 20

class ConvBNReLU(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 dialation=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size,
                      padding=padding * dialation,
                      dilation=dialation,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.net(x)
    
class RSU(nn.Module):
    def __init__(self, 
                 level,
                 in_channels, 
                 mid_channels,
                 out_channels):
        super().__init__()
        assert level >= 2, f"RSU Block should have a level >= 2, receiving {level}."
        
        self.ReBNConvIn = ConvBNReLU(in_channels, out_channels)
        
        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        self.encoders.append(ConvBNReLU(out_channels,
                                        mid_channels,))
        
        for _ in range(level - 1):
            self.encoders.append(ConvBNReLU(mid_channels,
                                            mid_channels))
            self.pools.append(nn.MaxPool2d(kernel_size=2,
                                           stride=2,
                                           ceil_mode=True)) # To ensure downsampling reversible
        
        # Bottleneck (Deepest Level)
        self.bottleneck = ConvBNReLU(mid_channels,
                                     mid_channels,
                                     dialation=2,
                                     padding=2)
        
        # Decoder
        self.decoders = nn.ModuleList()
        for _ in range(level - 1):
            self.decoders.append(ConvBNReLU(mid_channels * 2,
                                            mid_channels))
        
        self.ReBNConvOut = ConvBNReLU(mid_channels * 2,
                                      out_channels)
        
    def forward(self, x):
        hidden_x = self.ReBNConvIn(x)
        enc_feats = []
        hidden = hidden_x
        
        # Encode
        for i, enc in enumerate(self.encoders):
            hidden = enc(hidden)
            enc_feats.append(hidden)
            if i < len(self.pools):
                hidden = self.pools[i](hidden)
                
        # Bottleneck
        hidden = self.bottleneck(hidden)
        
        # Decode
        for i, dec in enumerate(self.decoders):
            # auto fit enc_feature size.
            hidden = nn.functional.interpolate(hidden,
                                             size=enc_feats[-(i+1)].shape[2:],
                                             mode='bilinear',
                                             align_corners=False)
            hidden = torch.cat([hidden, enc_feats[-(i+1)]], dim=1)
            hidden = dec(hidden)
        
        hidden = nn.functional.interpolate(hidden,
                                           size=enc_feats[0].shape[2:],
                                           mode='bilinear',
                                           align_corners=False) # ?
        hidden = self.ReBNConvOut(torch.cat([hidden, enc_feats[0]], dim=1))
        
        return hidden + hidden_x # Residual Connection
    
class RSU4F(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.ReBNConvIn = ConvBNReLU(in_channels, out_channels)
        self.ReBNConvE1 = ConvBNReLU(out_channels, mid_channels, dialation=1, padding=1)
        self.ReBNConvE2 = ConvBNReLU(mid_channels, mid_channels, dialation=2, padding=2)
        self.ReBNConvE3 = ConvBNReLU(mid_channels, mid_channels, dialation=4, padding=4)
        self.ReBNConvE4 = ConvBNReLU(mid_channels, mid_channels, dialation=8, padding=8)
        self.ReBNConvD3 = ConvBNReLU(mid_channels * 2, mid_channels, dialation=4, padding=4)
        self.ReBNConvD2 = ConvBNReLU(mid_channels * 2, mid_channels, dialation=2, padding=2)
        self.ReBNConvOut = ConvBNReLU(mid_channels * 2, out_channels, dialation=1, padding=1)
        
    def forward(self, x):
        hidden_x = self.ReBNConvIn(x)
        hidden_x1 = self.ReBNConvE1(hidden_x)
        hidden_x2 = self.ReBNConvE2(hidden_x1)
        hidden_x3 = self.ReBNConvE3(hidden_x2)
        hidden_x4 = self.ReBNConvE4(hidden_x3)
        hidden_x3d = self.ReBNConvD3(torch.cat([hidden_x3, hidden_x4], dim=1))
        hidden_x2d = self.ReBNConvD2(torch.cat([hidden_x2, hidden_x3d], dim=1))
        hidden_x1d = self.ReBNConvOut(torch.cat([hidden_x1, hidden_x2d], dim=1))
        return hidden_x + hidden_x1d
    
class SideHead(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 
                              out_channels, 
                              kernel_size=1)
    def forward(self, x, target_size):
        x = self.conv(x)
        return nn.functional.interpolate(x,
                                         size=target_size,
                                         mode='bilinear',
                                         align_corners=False)

class U2Net(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        # Encoders
        self.enc1 = RSU(7, in_channels, 32, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2 = RSU(6, 64, 32, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc3 = RSU(5, 128, 64, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc4 = RSU(4, 256, 128, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc5 = RSU4F(512, 256, 512)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc6 = RSU4F(512, 256, 512)
        
        # Decoders
        self.dec5 = RSU4F(512 * 2, 256, 512)
        self.dec4 = RSU(4, 512 * 2, 128, 256)
        self.dec3 = RSU(5, 256 * 2, 64, 128)
        self.dec2 = RSU(6, 128 * 2, 32, 64)
        self.dec1 = RSU(7, 64 * 2, 16, 64)        
        
        # SideHeads
        self.side1 = SideHead(64, out_channels)
        self.side2 = SideHead(64, out_channels)
        self.side3 = SideHead(128, out_channels)
        self.side4 = SideHead(256, out_channels)
        self.side5 = SideHead(512, out_channels)
        self.side6 = SideHead(512, out_channels)
        
        # Fusion
        self.outconv = nn.Conv2d(6 * out_channels, out_channels, kernel_size=1)
    
        # Init Weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        
        # Encoder
        hx1 = self.enc1(x)
        hx2 = self.enc2(self.pool1(hx1))
        hx3 = self.enc3(self.pool2(hx2))
        hx4 = self.enc4(self.pool3(hx3))
        hx5 = self.enc5(self.pool4(hx4))
        hx6 = self.enc6(self.pool5(hx5))
        
        # Decoder
        hx6up = nn.functional.interpolate(hx6, size=hx5.shape[2:], mode='bilinear', align_corners=False)
        hx5d  = self.dec5(torch.cat([hx6up, hx5], dim=1))

        hx5up = nn.functional.interpolate(hx5d, size=hx4.shape[2:], mode='bilinear', align_corners=False)
        hx4d  = self.dec4(torch.cat([hx5up, hx4], dim=1))

        hx4up = nn.functional.interpolate(hx4d, size=hx3.shape[2:], mode='bilinear', align_corners=False)
        hx3d  = self.dec3(torch.cat([hx4up, hx3], dim=1))

        hx3up = nn.functional.interpolate(hx3d, size=hx2.shape[2:], mode='bilinear', align_corners=False)
        hx2d  = self.dec2(torch.cat([hx3up, hx2], dim=1))

        hx2up = nn.functional.interpolate(hx2d, size=hx1.shape[2:], mode='bilinear', align_corners=False)
        hx1d  = self.dec1(torch.cat([hx2up, hx1], dim=1))

        # Side Outputs
        d1 = self.side1(hx1d, (H, W))
        d2 = self.side2(hx2d, (H, W))
        d3 = self.side3(hx3d, (H, W))
        d4 = self.side4(hx4d, (H, W))
        d5 = self.side5(hx5d, (H, W))
        d6 = self.side6(hx6,  (H, W))
        
        # Fusion Output
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
        
def bce_loss(pred, target):
    return nn.functional.binary_cross_entropy(pred, target, 'mean')

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
    
    model = U2Net(in_channels=3, out_channels=1)
    model.to(device)
    torch.compile(model)
    
    criterion_BCE = bce_loss
    criterion_Dice = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = torch.amp.GradScaler('cuda')
    
    # save model
    # Create a directory with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_dir = config.LOG_DIR / 'NUNet-MoNuSeg' / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        metric = Accumulator(3)
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)
            labels -= (labels == 2).int()
            images = tv_tensors.Image(images)
            labels = tv_tensors.Mask(labels)
            
            images, labels = sync_transforms(images, labels)
            images = image_transforms(images)
            labels = labels.squeeze(1).long()  # Convert (B, 1, H, W) to (B, H, W) for loss calculation
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                # Assuming labels are already class indices
                outputs = model(images)
                if isinstance(outputs, tuple):
                    # out: (B, 3, H, W)
                    # labels: (B, H, W)
                    loss_ce = sum(criterion_BCE(out, labels) for out in outputs) / len(outputs)
                    loss_dice = sum(criterion_Dice(out, labels) for out in outputs) / len(outputs)
                    loss = 0.4 * loss_ce + 0.6 * loss_dice
                    
                else:
                    loss_ce = criterion_BCE(outputs, labels)
                    loss_dice = criterion_Dice(outputs, labels)
                    loss = 0.4 * loss_ce + 0.6 * loss_dice

            scaler.scale(loss).backward()
            
            # loss.backward()
            scaler.unscale_(optimizer)
            
            # # print grad
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name}: grad norm = {param.grad.norm().item():.4f}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # optimizer.step()
            
            scaler.step(optimizer)
            scaler.update()
            
            with torch.no_grad():
                metric.add(loss_ce.item() * images.size(0), loss_dice.item() * images.size(0), images.size(0))
    
        lr_scheduler.step()
        print(f"Epoch {epoch+1}, CE Loss: {metric[0] / metric[2]:.4f}, Dice Loss: {metric[1] / metric[2]:.4f}")
    
        if epoch % 5 == 0:
            torch.save(model.state_dict(), save_dir / f"nested_unet_epoch_{epoch+1}.pth")
    
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
    