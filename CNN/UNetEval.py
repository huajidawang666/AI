import torch
import config
import cv2
from utils import CUDA
from CNN.NUNet import NestedUNet, image_transforms
import torch.nn.functional as F

device = CUDA.check_CUDA_available()

model = NestedUNet(in_channels=3, num_classes=3, deep_supervision=True)
state_dict = torch.load(config.LOG_DIR / 'NUNet-MoNuSeg' / 'latest.pth',
                        map_location=device)
model.load_state_dict(state_dict)
model.to(device)

# simple test
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
    pred_mask = F.one_hot(indices, num_classes=3) * 255.0
    cv2.imwrite(str(config.DATA_DIR / 'TCGA-18-5592-01Z-00-DX1_0_0_pred.png'), pred_mask.astype('uint8') * 255)