import config
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import logging
import torch
from torchvision import tv_tensors
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_PATH = config.DATA_DIR / 'MoNuSeg'
IMAGE_PATH = DATA_PATH / 'Tissue Images'
ANNOTATION_PATH = DATA_PATH / 'Annotations'
LABEL_PATH = DATA_PATH / 'masks'
PATCH_PATH = DATA_PATH / 'patches'

def xml_to_mask(xml_path, save_path, image_size=(1000, 1000)):
    inside_mask = np.zeros(image_size, dtype=np.uint8)
    boundary_mask = np.zeros(image_size, dtype=np.uint8)
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    for region in root.findall('.//Region'):
        vertices = []
        for vertex in region.findall('.//Vertex'):
            x = float(vertex.get('X'))
            y = float(vertex.get('Y'))
            x = np.clip(x, 0, image_size[1] - 1)
            y = np.clip(y, 0, image_size[0] - 1)
            vertices.append((x, y))
        
        if len(vertices) >= 3:
            pts = np.array([vertices], dtype=np.int32)
            cv2.fillPoly(inside_mask, pts, 255)
            # erode
            kernel = np.ones((5, 5), dtype=np.uint8)
            eroded_cell = cv2.erode(inside_mask, kernel, iterations=1)
            # boundary
            edge_cell = cv2.absdiff(inside_mask, eroded_cell)
            
            inside_mask = cv2.bitwise_or(inside_mask, eroded_cell)
            boundary_mask = cv2.bitwise_or(boundary_mask, edge_cell)
    
    inside_mask[boundary_mask == 255] = 0
    background_mask = 255 - cv2.bitwise_or(inside_mask, boundary_mask)
    mask = cv2.merge([background_mask, inside_mask, boundary_mask])
    
    logging.debug(f"Saving mask to {save_path}")
    cv2.imwrite(save_path, mask)

def process_annotations():
    logging.info("Starting annotation processing...")
    num_files = len(list(ANNOTATION_PATH.glob('*.xml')))
    num_processed = 0
    num_skipped = 0
    num_failed = 0
    for xml_file in ANNOTATION_PATH.glob('*.xml'):
        mask_file = LABEL_PATH / (xml_file.stem + '.png')
        if not mask_file.exists():
            try:
                logging.debug(f"Processing {xml_file.name}")
                xml_to_mask(xml_file, mask_file)
                num_processed += 1
            except Exception as e:
                logging.error(f"Failed to process {xml_file.name}: {e}")
                num_failed += 1
        else:
            logging.debug(f"Mask for {xml_file.name} already exists, skipping.")
            num_skipped += 1
    logging.info(f"Annotation processing completed. \n Total: {num_files}, Processed: {num_processed}, Skipped: {num_skipped}, Failed: {num_failed}")

def patch_image(image_path, mask_path, patch_size=(256, 256), stride=(128, 128)):
    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path))
    
    h, w = image.shape[:2] # height, width
    for y in range(0, h - patch_size[1] + 1, stride[1]):
        for x in range(0, w - patch_size[0] + 1, stride[0]):
            image_patch = image[y:y+patch_size[1], x:x+patch_size[0]]
            mask_patch = mask[y:y+patch_size[1], x:x+patch_size[0]]
            
            patch_file = PATCH_PATH / 'images' / f"{image_path.stem}_{y}_{x}.png"
            cv2.imwrite(patch_file, image_patch)
            patch_file = PATCH_PATH / 'masks' / f"{mask_path.stem}_{y}_{x}.png"
            cv2.imwrite(patch_file, mask_patch)           
    
def patching():
    logging.info("Starting patching process...")
    num_images = len(list(IMAGE_PATH.glob('*.tif')))
    num_patches = 0
    for image_file in IMAGE_PATH.glob('*.tif'):
        mask_file = LABEL_PATH / (image_file.stem + '.png')
        if mask_file.exists():
            logging.debug(f"Patching {image_file.name} and {mask_file.name}")
            patch_image(image_file, mask_file)
            num_patches += 1
        else:
            logging.warning(f"Mask for {image_file.name} not found, skipping patching.")
    logging.info(f"Patching completed. Total images: {num_images}, Patched: {num_patches}")

class MoNuSegDataset(Dataset):
    def __init__(self, patch_dir=PATCH_PATH, transform=None):
        self.image_paths = sorted((patch_dir / 'images').glob('*.png'))
        self.mask_paths = sorted((patch_dir / 'masks').glob('*.png'))
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = cv2.imread(str(self.image_paths[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.mask_paths[idx])) # mask shape: (H, W, 3), each channel is binary for background, inside, boundary
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        # deduce type explicitly
        image: np.ndarray
        mask: np.ndarray
        
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1).copy())
        label_tensor = torch.from_numpy(mask.transpose(2, 0, 1).argmax(axis=0, keepdims=True).astype(np.uint8).copy()) # Convert to (H, W) with class indices
        
        image = tv_tensors.Image(image_tensor)
        label = tv_tensors.Mask(label_tensor)
        
        if self.transform:
            image, label = self.transform(image, label)
        return image, label

if __name__ == "__main__":
    LABEL_PATH.mkdir(exist_ok=True, parents=True)
    IMAGE_PATH.mkdir(exist_ok=True, parents=True)
    (PATCH_PATH / 'images').mkdir(exist_ok=True, parents=True)
    (PATCH_PATH / 'masks').mkdir(exist_ok=True, parents=True)
    process_annotations()
    patching()
        
    