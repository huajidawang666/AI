import torch
import torch.nn.functional as F
import config
import cv2
from datetime import datetime
from torchvision import tv_tensors
from torch import nn
from torchvision.models import resnet50, resnet101, ResNet50_Weights, ResNet101_Weights
from torchvision.transforms import v2
from tqdm import tqdm
from dataset.MoNuSeg import MoNuSegDataset
from utils.metric import Accumulator

NUM_EPOCHS = 20

class SeparableConv2d(nn.Module):
    def __init__(self,
                 in_channels, 
                 out_channels, 
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels,
                                   in_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   dilation=dilation,
                                   groups=in_channels, # depthwise convolution
                                   bias=bias)
        self.pointwise = nn.Conv2d(in_channels,
                                   out_channels,
                                   kernel_size=1,
                                   bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ConvBNReLU(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1, 
                 padding=1,
                 dilation=1,
                 bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride,
                              padding,
                              dilation,
                              bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = ConvBNReLU(in_channels,
                               out_channels,
                               kernel_size=1,
                               padding=0)
    def forward(self, x):
        size = x.shape[-2:]
        x = self.pool(x)
        x = self.conv(x)
        return F.interpolate(x,
                             size=size, 
                             mode='bilinear',
                             align_corners=False)
        
class ASPP(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels=256,
                 atrous_rates=(6, 12, 18)):
        super().__init__()
        self.aspp_blocks = nn.ModuleList()
        self.aspp_blocks.append(ConvBNReLU(in_channels,
                                           out_channels,
                                           kernel_size=1,
                                           padding=0))
        for rate in atrous_rates:
            self.aspp_blocks.append(ConvBNReLU(in_channels,
                                               out_channels,
                                               kernel_size=3,
                                               padding=rate,
                                               dilation=rate))
        self.aspp_blocks.append(ASPPPooling(in_channels, out_channels))
        self.project = nn.Sequential(
            ConvBNReLU(len(self.aspp_blocks) * out_channels,
                       out_channels,
                       kernel_size=1,
                       padding=0),
            nn.Dropout(0.5)
        )
    def forward(self, x):
        out = [conv(x) for conv in self.aspp_blocks]
        out = torch.cat(out, dim=1)
        return self.project(out)
    
class ResNetBackbone(nn.Module):
    def __init__(self,
                 backbone='resnet50',
                 output_stride=16,
                 pretained=True):
        super().__init__()
        assert output_stride in (8, 16)
        assert backbone in ('resnet50', 'resnet101')
        
        weights = None
        resnet = None
        if backbone == 'resnet50':
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretained else None
            resnet = resnet50(weights=weights)
        elif backbone == 'resnet101':
            weights = ResNet101_Weights.IMAGENET1K_V1 if pretained else None
            resnet = resnet101(weights=weights)
            
        assert weights is not None, "Pretrained weights not found for the specified backbone."
        assert resnet is not None, "ResNet model could not be loaded."
        
        self.initial = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # Adjust strides and dilations for output_stride
        if output_stride == 16:
            self.layer4 = self._make_layer_dilated(self.layer4, stride=1, dilation=2)
        elif output_stride == 8:
            self.layer3 = self._make_layer_dilated(self.layer3, stride=1, dilation=2)
            self.layer4 = self._make_layer_dilated(self.layer4, stride=1, dilation=4)
    
    @staticmethod
    def _make_layer_dilated(layer, stride, dilation):
        for m in layer.modules():
            if isinstance(m, nn.Conv2d):
                if m.stride == (2, 2):
                    m.stride = (stride, stride)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilation, dilation)
                    m.padding = (dilation, dilation)
        return layer
    
    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        low_level_features = x  # Save low-level features for decoder
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return low_level_features, x
    
class Decoder(nn.Module):
    def __init__(self,
                 num_classes,
                 low_level_channels=256,
                 aspp_channels=256):
        super().__init__()
        self.low_level_conv = ConvBNReLU(low_level_channels,
                                         48,
                                         kernel_size=1,
                                         padding=0)
        self.conv1 = SeparableConv2d(aspp_channels + 48,
                                     256)
        self.conv2 = SeparableConv2d(256, 256)
        self.cls = nn.Conv2d(256, num_classes, kernel_size=1)
    
    def forward(self, low_level_features, aspp_output, input_size):
        low_level_features = self.low_level_conv(low_level_features)
        aspp_output = F.interpolate(aspp_output,
                                    size=low_level_features.shape[-2:],
                                    mode='bilinear',
                                    align_corners=False)
        x = torch.cat([low_level_features, aspp_output], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.cls(x)
        
        x = F.interpolate(x,
                          size=input_size,
                          mode='bilinear',
                          align_corners=False)
        return x

class DeepLabV3Plus(nn.Module):
    def __init__(self,
                 num_classes,
                 backbone='resnet50',
                 output_stride=16,
                 pretained=True):
        super().__init__()
        self.backbone = ResNetBackbone(backbone, output_stride, pretained)
        # ResNet gives a C=2048 output at layer4 and a C=256 output at layer1
        self.aspp = ASPP(in_channels=2048,
                         out_channels=256,
                         atrous_rates=(6, 12, 18) if output_stride == 16 else (12, 24, 36))
        self.decoder = Decoder(num_classes, 
                               low_level_channels=256, 
                               aspp_channels=256)
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        input_size = x.shape[-2:]
        low_level_features, backbone_output = self.backbone(x)
        aspp_output = self.aspp(backbone_output)
        return self.decoder(low_level_features, aspp_output, input_size)
    

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
    
    model = DeepLabV3Plus(num_classes=3, backbone='resnet50', output_stride=16, pretained=True)
    model.to(device)
    torch.compile(model)
    
    weights = torch.tensor([2.5, 1.0, 1.0]).to(device)
    criterion_CE = nn.CrossEntropyLoss(weight=weights)
    criterion_Dice = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
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
            
            images = tv_tensors.Image(images)
            labels = tv_tensors.Mask(labels)
            
            images, labels = sync_transforms(images, labels)
            images = image_transforms(images)
            labels = labels.squeeze(1).long()  # Convert (B, 1, H, W) to (B, H, W) for loss calculation
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                # Assuming labels are already class indices
                outputs = model(images)
                loss_ce = criterion_CE(outputs, labels)
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
    