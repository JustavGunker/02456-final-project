import os
import torch
import tifffile as tiff
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from pathlib import Path

class ImageMaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir, resize=(768, 768)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.resize = resize

        # List of image files
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.tif')])

        # Match masks by replacing 'image_v2_' with 'image_v2_mask_'
        mask_dir_files = os.listdir(mask_dir)
        self.mask_files = []
        for img_name in self.image_files:
            mask_name = img_name.replace('image_v2_', 'image_v2_mask_')
            if mask_name not in mask_dir_files:
                raise FileNotFoundError(f"Mask {mask_name} not found for image {img_name}")
            self.mask_files.append(mask_name)

        assert len(self.image_files) == len(self.mask_files), "Image and mask count mismatch!"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        # Load TIFF images
        image = tiff.imread(img_path)
        mask = tiff.imread(mask_path)

        # Contrast stretch image to 0-1
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        image = torch.from_numpy(image).float()

        # Convert mask to tensor
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        mask = ((mask < 4/255) | (mask == 118.005585/255) | (mask == 116.12109375/255) ).astype(np.float32)
        mask = torch.from_numpy(mask).float()

        # Add channel dimension to image if grayscale
        if image.ndim == 2:
            image = image.unsqueeze(0)  # C x H x W
        else:
            image = image.permute(2,0,1)  # HWC -> CHW

        # Resize image and mask
        image = F.interpolate(image.unsqueeze(0), size=self.resize, mode='bilinear', align_corners=False).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=self.resize, mode='nearest').squeeze(0).float()

        return image, mask
    
class unlabeledData(Dataset):
    def __init__(self, unlabeled_dir, resize=(768, 768)):
        self.unlabeled_dir = unlabeled_dir
        self.resize = resize
        self.crop_h, self.crop_w = (768, 768)
        self.unlabeled_files = sorted([f for f in os.listdir(unlabeled_dir) if f.endswith('.tif')])

    def __len__(self):
        return len(self.unlabeled_files)

    def __getitem__(self, idx):
        path = os.path.join(self.unlabeled_dir, self.unlabeled_files[idx])
        img = tiff.imread(path)

        # Convert to tensor and normalize to 0-1
        img = torch.from_numpy(img).float()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # Add channel dimension if grayscale
        if img.ndim == 2:
            img = img.unsqueeze(0)  # C x H x W

        # Crop center 768x768
        _, h, w = img.shape
        start_h = (h - self.crop_h) // 2
        start_w = (w - self.crop_w) // 2
        img = img[:, start_h:start_h + self.crop_h, start_w:start_w + self.crop_w]
        img = F.interpolate(img.unsqueeze(0), size=self.resize, mode='bilinear', align_corners=False).squeeze(0)

        return img