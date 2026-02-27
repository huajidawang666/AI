import torch
import math
from torchvision.models import resnet50
from torchvision.ops import nms, roi_align
from torch import nn
from typing import List, Tuple
from torch.nn import functional as F


class FPN(nn.Module):
    """
    Input:
        C2, C3, C4, C5 (From ResNet Backbone)
        C2: 256 channels, stride 4
        C3: 512 channels, stride 8
        C4: 1024 channels, stride 16
        C5: 2048 channels, stride 32
        
        Output: P2, P3, P4, P5
        P2: 256 channels, stride 4
        P3: 256 channels, stride 8
        P4: 256 channels, stride 16
        P5: 256 channels, stride 32
    """
    def __init__(self,
                 in_channels_list: list[int] = [256, 512, 1024, 2048],
                 out_channels: int = 256):
        super().__init__()
        self.out_channels = out_channels
        
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for in_channels in in_channels_list 
        ])
        
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in range(len(in_channels_list))
        ])
        
    def forward(self, 
                features_list: list[torch.Tensor]) -> list[torch.Tensor]:
        # Lateral connections
        lateral_features = [lateral_conv(feature)
                            for lateral_conv, feature in zip(self.lateral_convs, features_list)]
        
        # Top-down pathway
        for i in range(len(lateral_features) - 1, 0, -1):
            lateral_features[i - 1] += nn.functional.interpolate(
                lateral_features[i], size=lateral_features[i - 1].shape[-2:], mode="bilinear", align_corners=False
            )
        
        # Output convolutions
        output_features = [output_conv(feature)
                           for output_conv, feature in zip(self.output_convs, lateral_features)]
        
        return output_features
    
class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet50(pretrained=True)
        
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        self.fpn = FPN()
        
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        c1 = self.layer0(x)  # Output stride 4
        c2 = self.layer1(c1) # Output stride 4, ch=256
        c3 = self.layer2(c2) # Output stride 8, ch=512
        c4 = self.layer3(c3) # Output stride 16, ch=1024
        c5 = self.layer4(c4) # Output stride 32, ch=2048
        
        features_list = [c2, c3, c4, c5]
        fpn_features = self.fpn(features_list)
        
        return fpn_features
    
class AnchorGenerator(nn.Module):
    def __init__(self, 
                 sizes: list[int] = [32, 64, 128, 256],
                 aspect_ratios: list[float] = [0.5, 1.0, 2.0],
                 strides: list[int] = [4, 8, 16, 32]):
        super().__init__()
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.strides = strides
        
        self.relative_anchors = []
        for size in sizes:
            self.relative_anchors.append(
                self.generate_anchors(size, aspect_ratios)
            )    
    
    @staticmethod
    def generate_anchors(size: int, aspect_ratios: list[float]) -> torch.Tensor:
        size = torch.tensor(size, dtype=torch.float32)
        aspect_ratios = torch.tensor(aspect_ratios, dtype=torch.float32)
        
        h_ratio = torch.sqrt(aspect_ratios)
        w_ratio = 1 / h_ratio
        
        heights = size * h_ratio
        widths = size * w_ratio
        
        anchors = torch.stack([
            -widths / 2, -heights / 2, widths / 2, heights / 2
        ], dim=1)
        return anchors
    
    def forward(self, features_list: list[torch.Tensor]) -> list[torch.Tensor]:   
        """
        Return:
            (B=1, Num_Anchors*H*W, 4) for each feature map
            We set batch size to 1 for simplicity, as anchors are the same across the batch. (Using broadcasting to handle)
        """
        anchors_list = []
        for anchors, stride, feature in zip(self.relative_anchors, self.strides, features_list):
            anchors = anchors.to(feature.device) # (Num_Anchors, 4)
            
            batch_size, _, height, width = feature.shape
            
            grid_x = torch.arange(width) * stride
            grid_y = torch.arange(height) * stride
            
            grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
            
            grid = torch.stack([grid_x, grid_y, grid_x, grid_y], dim=-1).float().to(feature.device)  # (H, W, 4)
            
            anchors_grid = anchors.view(1, 1, 1, -1, 4) + grid.view(1, height, width, 1, 4)  # (B=1, H=1, W=1, Num_Anchors, Coord=4) + (B=1, H, W, Pixel=1, Coord=4) = (B=1, H, W, num_anchors, 4)
            anchors_grid = anchors_grid.view(1, -1, 4)  # (B=1, H*W*num_anchors, 4)
            anchors_list.append(anchors_grid)
        
        return anchors_list # (1, Num_Anchors*H*W, 4) for each feature map
    
class RPNHead(nn.Module):
    def __init__(self, in_channels: int = 256, num_anchors: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)
        
    def forward(self, features_list: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        cls_logits, bbox_pred = [], []
        for feature in features_list:    
            x = nn.functional.relu(self.conv(feature))
            cls_logits.append(self.cls_logits(x))
            bbox_pred.append(self.bbox_pred(x))
        return cls_logits, bbox_pred
    
class RPN(nn.Module):
    def __init__(self,
                 anchor_generator: AnchorGenerator | nn.Module = None,
                 rpn_head: RPNHead | nn.Module = None,
                 pre_nms_top_n: int = 2000,
                 post_nms_top_n: int = 1000,
                 nms_thresh: float = 0.7):
        super().__init__()
        self.anchor_generator = anchor_generator or AnchorGenerator()
        self.rpn_head = rpn_head or RPNHead()
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
    
    @staticmethod
    def apply_deltas(
        anchors: torch.Tensor,
        bbox_pred: torch.Tensor
    ):
        """
        Input:
            anchors: (B=1, Num_Anchors*H*W, 4) for each feature map
            bbox_pred: (B, Num_Anchors*4, H, W) for each feature map
        """
        
        widths = anchors[..., 2] - anchors[..., 0]
        heights = anchors[..., 3] - anchors[..., 1]
        ctr_x = anchors[..., 0] + 0.5 * widths
        ctr_y = anchors[..., 1] + 0.5 * heights
        
        dx = bbox_pred[..., 0]
        dy = bbox_pred[..., 1]
        dw = bbox_pred[..., 2].clamp(max=math.log(1000.0 / 16.0))  # Prevent overflow
        dh = bbox_pred[..., 3].clamp(max=math.log(1000.0 / 16.0))  # Prevent overflow    
        
        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = widths * torch.exp(dw)
        pred_h = heights * torch.exp(dh)
        
        pred_boxes = torch.stack([
            pred_ctr_x - 0.5 * pred_w,
            pred_ctr_y - 0.5 * pred_h,
            pred_ctr_x + 0.5 * pred_w,
            pred_ctr_y + 0.5 * pred_h
        ], dim=-1) # (B, H*W*Num_Anchors, 4)
        
        return pred_boxes
    
    @staticmethod
    def clip_boxes(boxes: torch.Tensor,
                   image_size: Tuple[int, int],
                   clip_margin: int = 0):
        H, W = image_size
        x1 = boxes[..., 0].clamp(min=clip_margin, max=W - clip_margin)
        y1 = boxes[..., 1].clamp(min=clip_margin, max=H - clip_margin)
        x2 = boxes[..., 2].clamp(min=clip_margin, max=W - clip_margin)
        y2 = boxes[..., 3].clamp(min=clip_margin, max=H - clip_margin)
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)
        return boxes
        
    @staticmethod
    def serial_nms(
        boxes: torch.Tensor,
        scores: torch.Tensor,
        iou_threshold: float = 0.7,
        max_output: int = 1000):
        
        final_proposals = []
        final_batch_indices = []
        for idx in range(boxes.shape[0]):
            batch_boxes = boxes[idx] # (N, 4) 
            batch_scores = scores[idx] # (N, 1)
            
            keep = nms(batch_boxes, batch_scores.squeeze(-1), iou_threshold)
            
            keep = keep[:max_output]
            final_proposals.append(batch_boxes[keep])
            final_batch_indices.append(torch.full((len(keep),), idx, dtype=torch.float32, device=boxes.device))
            
        final_proposals = torch.cat(final_proposals, dim=0)
        final_batch_indices = torch.cat(final_batch_indices, dim=0).float()
        
        return torch.cat([final_batch_indices.unsqueeze(-1), final_proposals], dim=-1)
        
        
    @staticmethod
    def batched_nms(
        boxes: torch.Tensor, # (B, N, 4)
        scores: torch.Tensor, # (B, N)
        iou_threshold: float = 0.7,
        max_output: int = 1000
    ):
        return
        
    def forward(self,
                features_list: list[torch.Tensor],
                image_size: Tuple[int, int]) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:        
        anchors_list = self.anchor_generator(features_list) # list of (1, Num_Anchors*H*W, 4) for each feature map
        cls_logits, bbox_pred = self.rpn_head(features_list) # list of (B, Num_Anchors, H, W) for each feature map, (B, Num_Anchors*4, H, W) for each feature map
        
        all_anchors = torch.cat(anchors_list, dim=1) # (B=1, Total_Num_Anchors, 4)
        all_cls_logits = torch.cat([logit.permute(0, 2, 3, 1).reshape(logit.shape[0], -1, 1) for logit in cls_logits], dim=1) # (B, Total_Num_Anchors, 1)
        all_bbox_pred = torch.cat([bbox.permute(0, 2, 3, 1).reshape(bbox.shape[0], -1, 4) for bbox in bbox_pred], dim=1) # (B, Total_Num_Anchors, 4)
        
        
        all_cls_logits = F.sigmoid(all_cls_logits) # (B, Total_Num_Anchors, 1)
        topk_n = min(self.pre_nms_top_n, all_cls_logits.shape[1])
        topk_scores, topk_indices = torch.topk(all_cls_logits, k=topk_n, dim=1) # (B, topk_n, 1)
        
        all_anchors = all_anchors.expand(all_cls_logits.shape[0], -1, -1) # (B, Total_Num_Anchors, 4)
        topk_anchors = torch.gather(all_anchors, dim=1, index=topk_indices.expand(-1, -1, 4)) # (B, topk_n, 4)
        topk_bbox_pred = torch.gather(all_bbox_pred, dim=1, index=topk_indices.expand(-1, -1, 4)) # (B, topk_n, 4)
        
        proposals = self.apply_deltas(topk_anchors, topk_bbox_pred) # (B, topk_n, 4)
        proposals = self.clip_boxes(proposals, image_size=image_size)
        
        # NMS
        proposals = self.serial_nms(
            proposals, topk_scores, iou_threshold=self.nms_thresh, max_output=self.post_nms_top_n
        ) # (Total_Kept, 5) (batch_index, x1, y1, x2, y2)
        
        return proposals
        
class ROIAlign(nn.Module):
    def __init__(self, output_size: Tuple[int, int] = (7, 7), out_channels: int = 256):
        super().__init__()
        self.output_size = output_size
        self.out_channels = out_channels
    
    @staticmethod
    def assign_level(boxes, k0=4):
        """
        Assign each box to a level in the FPN based on its size.
        Input:
            boxes: (N, 5) in (batch_idx, x1, y1, x2, y2)
        Output:
            levels: (N,) with values in {0, 1, 2, 3} corresponding to P2, P3, P4, P5
        """
        w = boxes[:, 3] - boxes[:, 1]
        h = boxes[:, 4] - boxes[:, 2]
        s = torch.sqrt(w * h).clamp(min=1e-6) # Avoid log(0)
        levels = torch.floor(k0 - 2 + torch.log2(s / 224.0)).clamp(0, 3).long()
        return levels
    
    def forward(self,
                features_list: list[torch.Tensor], 
                proposals: torch.Tensor,
                strides: List[int] = [4, 8, 16, 32]) -> torch.Tensor:
        """
        Input:
            proposals: (Total_Proposals, 5) in (batch_idx, x1, y1, x2, y2)
        Output:
            roi_features: (Total_Proposals, C, output_size[0], output_size[1])
        """
        Num_proposals = proposals.shape[0]
        output = torch.zeros((Num_proposals, self.out_channels, self.output_size[0], self.output_size[1]), device=proposals.device)
        
        levels = self.assign_level(proposals) # (Total_Proposals,)
        
        for level in range(4):
            level_indices = (levels == level).nonzero(as_tuple=True)[0]
            if len(level_indices) == 0:
                continue
            
            level_proposals = proposals[level_indices] # (Num_Level_Proposals, 5)
            
            pooled = roi_align(
                input=features_list[level], # (B, C, H, W)
                boxes=level_proposals, # (Num_Level_Proposals, 5)
                output_size=self.output_size,
                spatial_scale=1.0 / (strides[level]) # P2: 1/4, P3: 1/8, P4: 1/16, P5: 1/32
            ) # (Num_Level_Proposals, C, output_size[0], output_size[1])
            
            output[level_indices] = pooled
            
        return output, proposals[:, 0].long() # (Total_Proposals, C, output_size[0], output_size[1]), (Total_Proposals,) batch indices
    
class BoxHead(nn.Module):
    def __init__(self, in_channels: int = 256, num_classes: int = 80):
        super().__init__()
        self.fc1 = nn.Linear(in_channels * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * 4)
        
    def forward(self, roi_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = roi_features.view(roi_features.shape[0], -1) # (Total_Proposals, C*7*7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        cls_logits = self.cls_score(x) # (Total_Proposals, num_classes)
        bbox_pred = self.bbox_pred(x) # (Total_Proposals, num_classes*4)
        
        return cls_logits, bbox_pred
    
class MaskHead(nn.Module):
    def __init__(self, in_channels: int = 256, num_classes: int = 80):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.deconv = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.mask_pred = nn.Conv2d(256, num_classes, kernel_size=1)
        
    def forward(self, roi_features: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(roi_features))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.deconv(x))
        mask_logits = self.mask_pred(x) # (Total_Proposals, num_classes, (size * 2), (size * 2))
        return mask_logits

class MaskRCNN(nn.Module):
    def __init__(self, 
                 num_classes: int = 80,
                 training=True):
        super().__init__()
        self.backbone = Backbone()
        self.RPN = RPN()
        self.box_roi_align = ROIAlign()
        self.mask_roi_align = ROIAlign(output_size=(14, 14), out_channels=256)
        self.box_head = BoxHead(num_classes=num_classes)
        self.mask_head = MaskHead(num_classes=num_classes)
        self.training = training
    
    @staticmethod
    def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor):
        """
        Compute IoU between two sets of boxes.
        Input:
            boxes1: (N, 4) in (x1, y1, x2, y2)
            boxes2: (M, 4) in (x1, y1, x2, y2)
        Output:
            iou: (N, M) IoU matrix
        """
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2]) # (N, M, 2)
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) # (N, M, 2)
        
        wh = (rb - lt).clamp(min=0) # (N, M, 2)
        inter_area = wh[..., 0] * wh[..., 1]
        
        iou = inter_area / (area1[:, None] + area2 - inter_area) # (N, M)
        return iou
    
    @staticmethod
    def encode_boxes(proposals: torch.Tensor, gt_boxes: torch.Tensor):
        """
        Encode the gt_boxes relative to the proposals for box regression.
        Input:
            proposals: (N, 4) in (x1, y1, x2, y2)
            gt_boxes: (N, 4) in (x1, y1, x2, y2)
        Output:
            encoded: (N, 4) in (dx, dy, dw, dh)
        """
        pw = proposals[:, 2] - proposals[:, 0]
        ph = proposals[:, 3] - proposals[:, 1]
        pcx = proposals[:, 0] + 0.5 * pw
        pcy = proposals[:, 1] + 0.5 * ph
        
        gw = gt_boxes[:, 2] - gt_boxes[:, 0]
        gh = gt_boxes[:, 3] - gt_boxes[:, 1]
        gcx = gt_boxes[:, 0] + 0.5 * gw
        gcy = gt_boxes[:, 1] + 0.5 * gh
        
        dx = (gcx - pcx) / pw
        dy = (gcy - pcy) / ph
        dw = torch.log(gw / pw)
        dh = torch.log(gh / ph)
        
        return torch.stack([dx, dy, dw, dh], dim=1) # (N, 4)
        
        
    def assign_and_sample_proposals(self,
                                    proposals: torch.Tensor, # (Total, 5)
                                    targets: List[dict],
                                    num_samples: int = 512,
                                    pos_fraction: float = 0.25,
                                    pos_iou_thresh: float = 0.5,
                                    neg_iou_thresh: float = 0.5):
        """
        Input: 
            proposals: (Total_Proposals, 5) in (batch_idx, x1, y1, x2, y2)
            targets: List of dicts containing 'boxes' 'labels' and 'masks' for each image in the batch
            num_samples: Total number of samples to draw
            pos_fraction: Fraction of positive samples
            pos_iou_thresh: IoU threshold for positive samples
            neg_iou_thresh: IoU threshold for negative samples
        """
        num_pos = int(num_samples * pos_fraction)
        num_neg = num_samples - num_pos
        
        all_proposals = []
        all_gt_classes = []
        all_gt_encoded = []
        all_pos_mask = []
        all_sampled_gt_idx = []
        
        for batch_idx, target in enumerate(targets):
            gt_boxes = target['boxes'].to(proposals.device) # (Num_GT, 4)
            gt_labels = target['labels'].to(proposals.device) # (Num_GT,)
            
            batch_proposals = proposals[proposals[:, 0] == batch_idx][:, 1:] # (Num_Batch_Proposals, 4)
            
            if batch_proposals.shape[0] == 0:
                continue
            
            iou = self.box_iou(batch_proposals, gt_boxes) # (Num_Batch_Proposals, Num_GT)
            max_iou, max_iou_indices = iou.max(dim=1) # (Num_Batch_Proposals,)
            
            labels = torch.zeros(batch_proposals.shape[0], dtype=torch.long, device=proposals.device) # (Num_Batch_Proposals,)
            
            pos_mask = max_iou >= pos_iou_thresh
            labels[pos_mask] = gt_labels[max_iou_indices[pos_mask]]
            
            best_proposal_per_gt = iou.argmax(dim=0) # (Num_GT,)
            labels[best_proposal_per_gt] = gt_labels
            
            ignore_mask = (max_iou >= 0.4) & (max_iou < pos_iou_thresh)
            ignore_mask[best_proposal_per_gt] = False
            labels[ignore_mask] = -1
            
            pos_indices = (labels > 0).nonzero(as_tuple=True)[0]
            neg_indices = (labels == 0).nonzero(as_tuple=True)[0]
            
            if len(pos_indices) > num_pos:
                pos_indices = pos_indices[torch.randperm(len(pos_indices))[:num_pos]]
                
            actual_neg = num_samples - len(pos_indices)
            if len(neg_indices) > actual_neg:
                neg_indices = neg_indices[torch.randperm(len(neg_indices), device=proposals.device)[:actual_neg]]
                
            sampled_indices = torch.cat([pos_indices, neg_indices], dim=0)
            matched_gt_boxes = gt_boxes[max_iou_indices[sampled_indices]] # (Num_Sampled, 4)
            sampled_boxes = batch_proposals[sampled_indices] # (Num_Sampled, 4)
            gt_encoded = self.encode_boxes(sampled_boxes, matched_gt_boxes)
            
            sampled_proposals = torch.cat([torch.full((len(sampled_indices), 1), float(batch_idx), device=proposals.device), sampled_boxes], dim=1) # (Num_Sampled, 5)
            
            pos_mask = labels[sampled_indices] > 0
            
            all_proposals.append(sampled_proposals)
            all_gt_classes.append(labels[sampled_indices])
            all_gt_encoded.append(gt_encoded)
            all_pos_mask.append(pos_mask)
            all_sampled_gt_idx.append(max_iou_indices[sampled_indices])
            
        return (
            torch.cat(all_proposals, dim=0), 
            torch.cat(all_gt_classes, dim=0), 
            torch.cat(all_gt_encoded, dim=0), 
            torch.cat(all_pos_mask, dim=0),
            torch.cat(all_sampled_gt_idx, dim=0)
        )

    @staticmethod
    def assign_gt_masks(positive_proposals: torch.Tensor,
                        gt_idx: torch.Tensor,
                        targets: List[dict],
                        mask_size: Tuple[int, int] = (28, 28)):
        gt_masks_list = []
        gt_mask_labels_list = []
        
        for idx in range(len(positive_proposals)):
            batch_idx = int(positive_proposals[idx, 0].item())
            x1, y1, x2, y2 = positive_proposals[idx, 1:].int()
            
            matched_idx = gt_idx[idx].item()
            gt_mask = targets[batch_idx]['masks'][matched_idx] # (H, W)
            gt_label = targets[batch_idx]['labels'][matched_idx]
            
            H, W = gt_mask.shape
            x1 = x1.clamp(0, W - 1)
            y1 = y1.clamp(0, H - 1)
            x2 = x2.clamp(x1 + 1, W) # Ensure at least 1 pixel width
            y2 = y2.clamp(y1 + 1, H) # Ensure at least 1 pixel height
            
            cropped_mask = gt_mask[y1:y2, x1:x2].float() # (crop_h, crop_w)
            
            resized_mask = F.interpolate(cropped_mask.unsqueeze(0).unsqueeze(0), size=mask_size, mode='bilinear', align_corners=False).squeeze(0).squeeze(0) # (mask_size[0], mask_size[1])
            
            gt_masks_list.append(resized_mask>0.5) # Binarize the mask
            gt_mask_labels_list.append(gt_label)
            
        gt_masks = torch.stack(gt_masks_list, dim=0) # (Num_Positive, mask_size[0], mask_size[1])
        gt_mask_labels = torch.stack(gt_mask_labels_list, dim=0) # (Num_Positive,)
        
        return gt_masks, gt_mask_labels
    
    @staticmethod
    def postprocess_detections(cls_logits: torch.Tensor,
                                bbox_pred: torch.Tensor,
                                proposals: torch.Tensor,
                                image_size: Tuple[int, int],
                                score_thresh: float = 0.5,
                                nms_thresh: float = 0.5,
                                max_detections: int = 100):
            """
            Post-process the raw outputs from the box head to get final detections.
            Input:
                cls_logits: (Total_Proposals, num_classes)
                bbox_pred: (Total_Proposals, num_classes*4)
                proposals: (Total_Proposals, 5) in (batch_idx, x1, y1, x2, y2)
            Output:
                final_boxes: (Num_Detections, 5) in (batch_idx, x1, y1, x2, y2)
                final_labels: (Num_Detections,) class labels for each detection
                final_scores: (Num_Detections,) confidence scores for each detection
            """
            num_classes = cls_logits.shape[1]
            scores = F.softmax(cls_logits, dim=-1)  # (K, num_classes)

            # apply delta（复用 RPN 里的 apply_deltas）
            boxes = RPN.apply_deltas(
                proposals[:, 1:],                           # (K, 4)
                bbox_pred.view(-1, num_classes, 4)          # (K, num_classes, 4)
                # 注意：这里需要对每个类别分别 apply，可以用 proposal 广播
            )  # TODO: 按类别 apply deltas，下面是简化版

            all_boxes, all_scores, all_labels, all_batch = [], [], [], []

            batch_indices = proposals[:, 0].long()
            for cls_idx in range(1, num_classes):   # 跳过背景类 0
                cls_scores = scores[:, cls_idx]     # (K,)
                keep = cls_scores > score_thresh
                if keep.sum() == 0:
                    continue

                kept_boxes   = proposals[keep, 1:]  # 先用 proposal 坐标，后续补 apply_deltas
                kept_scores  = cls_scores[keep]
                kept_batch   = batch_indices[keep]

                # 按图分别做 NMS（不同图的框不应互相抑制）
                for b in kept_batch.unique():
                    b_mask = kept_batch == b
                    keep_nms = nms(kept_boxes[b_mask], kept_scores[b_mask], nms_thresh)
                    keep_nms = keep_nms[:max_detections]

                    all_boxes.append(kept_boxes[b_mask][keep_nms])
                    all_scores.append(kept_scores[b_mask][keep_nms])
                    all_labels.append(torch.full((keep_nms.shape[0],), cls_idx, device=cls_logits.device))
                    all_batch.append(torch.full((keep_nms.shape[0],), b.item(), device=cls_logits.device))

            if len(all_boxes) == 0:
                empty = torch.zeros(0, device=cls_logits.device)
                return empty, empty, empty.long(), empty.long()

            return (torch.cat(all_boxes),
                    torch.cat(all_scores),
                    torch.cat(all_labels),
                    torch.cat(all_batch))
    
        
    def forward(self, 
                images: torch.Tensor,
                image_size: Tuple[int, int], 
                targets: List[dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_size = images.shape[-2:]
        
        features_list = self.backbone(images)  # List of feature maps from FPN
        proposals = self.RPN(features_list, image_size)  # (Total_Proposals, 5) in (batch_idx, x1, y1, x2, y2)
        
        if self.training:
            sampled_proposals, gt_classes, gt_encoded, pos_mask, sampled_gt_idx = self.assign_and_sample_proposals(proposals, targets)
            # sampled_proposals: (Num_Sampled, 5) in (batch_idx, x1, y1, x2, y2)
            # gt_classes: (Num_Sampled,) class labels for each sampled proposal
            # gt_encoded: (Num_Sampled, 4) encoded box regression targets for each sampled proposal
            # pos_mask: (Num_Sampled,) boolean mask indicating which proposals are positive
            
            roi_features, _ = self.box_roi_align(features_list, sampled_proposals) # (Num_Sampled, C, 7, 7), (Num_Sampled,) batch indices
            
            cls_logits, bbox_pred = self.box_head(roi_features) # (Num_Sampled, num_classes), (Num_Sampled, num_classes*4)
            
            positive_proposals = sampled_proposals[pos_mask]
            
            mask_roi_features, _ = self.mask_roi_align(features_list, positive_proposals) # (Num_Positive, C, 14, 14), (Num_Positive,) batch indices
            
            mask_logits = self.mask_head(mask_roi_features) # (Num_Positive, num_classes, 28, 28)
            
            gt_masks, gt_mask_labels = self.assign_gt_masks(positive_proposals, sampled_gt_idx[pos_mask], targets) # (Num_Positive, 28, 28), (Num_Positive,)
        
            return {
                'cls_logits': cls_logits,
                'bbox_pred': bbox_pred,
                'gt_classes': gt_classes,
                'gt_encoded': gt_encoded,
                'mask_logits': mask_logits,
                'gt_masks': gt_masks,
                'gt_mask_labels': gt_mask_labels
            }
        else:
            roi_features, batch_indices = self.box_roi_align(features_list, proposals) # (Total_Proposals, C, 7, 7), (Total_Proposals,) batch indices
            
            cls_logits, bbox_pred = self.box_head(roi_features) # (Total_Proposals, num_classes), (Total_Proposals, num_classes*4)
            
            final_boxes, final_scores, final_labels, final_batch = \
                self.postprocess_detections(cls_logits, bbox_pred, proposals, image_size)
            # final_boxes: (N, 4)

            if final_boxes.shape[0] == 0:
                return []

            # ---- 6I. 重新构建 (K,5) 格式送 Mask Head ----
            final_proposals = torch.cat([
                final_batch.float().unsqueeze(1),
                final_boxes
            ], dim=1)  # (N, 5)

            mask_roi_features, _ = self.mask_roi_align(features_list, final_proposals)
            # mask_roi_features: (N, 256, 14, 14)

            # ---- 7I. Mask Head ----
            mask_logits = self.mask_head(mask_roi_features)
            # mask_logits: (N, num_classes, 28, 28)

            # 只取预测类别对应的那张 mask
            pred_masks = mask_logits[
                torch.arange(len(final_labels), device=mask_logits.device),
                final_labels
            ]  # (N, 28, 28)
            pred_masks = torch.sigmoid(pred_masks) > 0.5

            # 按图组织输出
            results = []
            B = images.shape[0]
            for b in range(B):
                b_mask = final_batch == b
                results.append({
                    'boxes'  : final_boxes[b_mask],    # (n, 4)
                    'scores' : final_scores[b_mask],   # (n,)
                    'labels' : final_labels[b_mask],   # (n,)
                    'masks'  : pred_masks[b_mask],     # (n, 28, 28)
                })
            return results
        
def compute_loss(outputs: dict) -> dict:
    cls_logits     = outputs['cls_logits']      # (Num_Sampled, num_classes)
    bbox_pred      = outputs['bbox_pred']       # (Num_Sampled, num_classes*4)
    gt_classes     = outputs['gt_classes']      # (Num_Sampled,)
    gt_encoded     = outputs['gt_encoded']      # (Num_Sampled, 4)
    mask_logits    = outputs['mask_logits']     # (Num_Pos, num_classes, 28, 28)
    gt_masks       = outputs['gt_masks']        # (Num_Pos, 28, 28)
    gt_mask_labels = outputs['gt_mask_labels']  # (Num_Pos,)

    loss_cls = F.cross_entropy(cls_logits, gt_classes)

    pos_mask = gt_classes > 0  # (Num_Sampled,)
    
    if pos_mask.sum() > 0:
        pos_cls = gt_classes[pos_mask]  # (Num_Pos,)

        bbox_pred_pos = bbox_pred[pos_mask]  # (Num_Pos, num_classes*4)
        bbox_pred_pos = bbox_pred_pos.view(-1, cls_logits.shape[1], 4)
        # (Num_Pos, num_classes, 4)

        bbox_pred_pos = bbox_pred_pos[
            torch.arange(len(pos_cls), device=pos_cls.device),
            pos_cls
        ]  # (Num_Pos, 4)

        loss_box = F.smooth_l1_loss(bbox_pred_pos, gt_encoded[pos_mask])
    else:
        loss_box = bbox_pred.sum() * 0 

    if len(gt_mask_labels) > 0:
        mask_logits_selected = mask_logits[
            torch.arange(len(gt_mask_labels), device=gt_mask_labels.device),
            gt_mask_labels
        ]  # (Num_Pos, 28, 28)

        loss_mask = F.binary_cross_entropy_with_logits(
            mask_logits_selected,
            gt_masks.float()
        )
    else:
        loss_mask = mask_logits.sum() * 0

    return {
        'loss_cls' : loss_cls,
        'loss_box' : loss_box,
        'loss_mask': loss_mask,
        'loss_total': loss_cls + loss_box + loss_mask
    }
        
import os
from torch.utils.data import DataLoader
from CNN.MaskRCNNSimple import MoNuSegDataset, MoNuSegInstanceDataset

def collate_fn(batch):
    """过滤掉 dataset 返回 None 的样本（无实例的图片）"""
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None
    images, targets = zip(*batch)
    return torch.stack(images, dim=0), list(targets)


def train_one_epoch(model, optimizer, dataloader, device, epoch):
    model.train()
    
    total_loss_cls  = 0.0
    total_loss_box  = 0.0
    total_loss_mask = 0.0
    total_loss      = 0.0
    num_batches     = 0

    for batch_idx, batch in enumerate(dataloader):
        if batch is None:
            continue

        images, targets = batch
        images  = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        outputs = model(images, (256, 256), targets=targets)
        losses  = compute_loss(outputs)

        losses['loss_total'].backward()
        
        # 梯度裁剪，防止训练早期梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        total_loss_cls  += losses['loss_cls'].item()
        total_loss_box  += losses['loss_box'].item()
        total_loss_mask += losses['loss_mask'].item()
        total_loss      += losses['loss_total'].item()
        num_batches     += 1

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | "
                  f"cls: {losses['loss_cls'].item():.4f}  "
                  f"box: {losses['loss_box'].item():.4f}  "
                  f"mask: {losses['loss_mask'].item():.4f}  "
                  f"total: {losses['loss_total'].item():.4f}")

    if num_batches == 0:
        print(f"Epoch {epoch}: no valid batches")
        return {}

    return {
        'loss_cls'  : total_loss_cls  / num_batches,
        'loss_box'  : total_loss_box  / num_batches,
        'loss_mask' : total_loss_mask / num_batches,
        'loss_total': total_loss      / num_batches,
    }


def train(num_epochs=10, batch_size=2, lr=0.005, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    # ---- 数据集 ----
    # 假设你已经有 base_train_dataset 和 base_val_dataset
    base_dataset = MoNuSegDataset()
    train_dataset = MoNuSegInstanceDataset(base_dataset, min_area=50)
    train_loader  = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2
    )

    # ---- 模型 ----
    model = MaskRCNN(num_classes=2)  # 1类细胞核 + 背景
    model.to(device)

    # ---- 优化器 ----
    # Backbone 用更小的 lr（预训练权重）
    backbone_params = list(model.backbone.parameters())
    head_params     = (list(model.RPN.parameters()) +
                       list(model.box_roi_align.parameters()) +
                       list(model.mask_roi_align.parameters()) +
                       list(model.box_head.parameters()) +
                       list(model.mask_head.parameters()))

    optimizer = torch.optim.SGD([
        {'params': backbone_params, 'lr': lr * 0.1},
        {'params': head_params,     'lr': lr},
    ], momentum=0.9, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # ---- 训练循环 ----
    os.makedirs('checkpoints', exist_ok=True)
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{num_epochs}")
        
        epoch_losses = train_one_epoch(model, optimizer, train_loader, device, epoch)
        scheduler.step()

        if epoch_losses:
            print(f"Epoch {epoch} avg | "
                  f"cls: {epoch_losses['loss_cls']:.4f}  "
                  f"box: {epoch_losses['loss_box']:.4f}  "
                  f"mask: {epoch_losses['loss_mask']:.4f}  "
                  f"total: {epoch_losses['loss_total']:.4f}")

        # 保存 checkpoint
        torch.save({
            'epoch'     : epoch,
            'model'     : model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'scheduler' : scheduler.state_dict(),
            'losses'    : epoch_losses,
        }, f'checkpoints/epoch_{epoch}.pth')


if __name__ == '__main__':
    train(num_epochs=10, batch_size=2, lr=0.005)