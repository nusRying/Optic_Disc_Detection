import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class RetinalDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, img_size=(512, 512)):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.img_size = img_size
        
        # Filter for valid image files
        valid_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
        self.images = [f for f in os.listdir(images_dir) if os.path.splitext(f)[1].lower() in valid_extensions]
        
        # Verify masks exist (assuming same filename or simple convention)
        # This might need adjustment based on specific dataset (DRIVE/STARE) naming conventions
        # For now, we assume mask has same name as image
        self.valid_images = []
        for img_name in self.images:
             mask_path = os.path.join(self.masks_dir, img_name)
             # Also check for .gif which is common for masks in some datasets
             if os.path.exists(mask_path) or os.path.exists(os.path.splitext(mask_path)[0] + '.gif'):
                 self.valid_images.append(img_name)
        
        self.images = self.valid_images
        print(f"Found {len(self.images)} matched image/mask pairs.")

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def __len__(self):
        return len(self.images)

    def preprocess(self, image):
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L-channel
        cl = self.clahe.apply(l)
        
        # Merge channels
        limg = cv2.merge((cl, a, b))
        
        # Convert back to RGB
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        
        return final

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        # Mask might be .gif or same extension
        mask_path = os.path.join(self.masks_dir, img_name)
        if not os.path.exists(mask_path):
             mask_path = os.path.splitext(mask_path)[0] + '.gif'

        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply CLAHE
        image = self.preprocess(image)
        
        # Load mask
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask)
        
        # Resize
        image = cv2.resize(image, self.img_size)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        
        # Normalize and convert to tensor
        image = image / 255.0
        mask = mask / 255.0
        
        # Threshold mask to be binary (0 or 1)
        mask = (mask > 0.5).astype(np.float32)
        
        # Channel first for PyTorch
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        mask = np.expand_dims(mask, axis=0).astype(np.float32)
        
        return torch.from_numpy(image), torch.from_numpy(mask)
