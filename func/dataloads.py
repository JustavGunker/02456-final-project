import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import os
import sys
import glob
import itertools
from pathlib import Path
import scipy.ndimage as ndimage
import torchio as tio


INPUT_SHAPE = (64, 64, 64) # ( D, H, W)
NUM_CLASSES = 3  # Background, Segment 1, Segment 2

class LiverDataset_aug(Dataset):
    """
    Dataset for Labeled 3D Liver Scans (e.g., imagesTr, labelsTr)
    
    Includes:
    - 3D Augmentations (via torchio)
    - CT Windowing (Clipping)
    - [0, 1] Normalization
    """
    def __init__(self, image_dir, label_dir, target_size=(128, 128, 128), augment=False):
        """
        Args:
            image_dir (Path): Path to the directory containing images (e.g., imagesTr).
            label_dir (Path): Path to the directory containing labels (e.g., labelsTr).
            target_size (tuple): The desired (D, H, W) to resize/pad to.
            augment (bool): Whether to apply 3D augmentations.
                             Set True for training, False for validation.
        """
        self.image_dir = image_dir / "imagesTr"
        self.label_dir = label_dir / "labelsTr"
        self.target_size = target_size
        self.augment = augment

        # Find all matching image/label pairs
        self.image_files = sorted([f for f in self.image_dir.glob("*.nii")])
        self.label_files = sorted([f for f in self.label_dir.glob("*.nii")])

        # Basic validation
        if len(self.image_files) == 0 or len(self.label_files) == 0:
            raise FileNotFoundError(f"No .nii files found in {self.image_dir} or {self.label_dir}")
        if len(self.image_files) != len(self.label_files):
            raise ValueError("Mismatch in number of images and labels.")
            
        print(f"Found {len(self.image_files)} image/label pairs.")

        # The self.augment flag will be checked in __getitem__
        self.transform = tio.Compose([
            tio.RandomFlip(axes=('LR', 'AP', 'IS')), # Randomly flip on any axis
            tio.RandomAffine(
                scales=(0.9, 1.2),  # Random zoom (90% to 120%)
                degrees=15,         # Random rotation up to 15 degrees
                isotropic=True,     # Ensure scaling is the same in all dims
                image_interpolation='linear',
                label_interpolation='nearest' # Use nearest neighbor for masks
            ),
            tio.RandomElasticDeformation(
                num_control_points=7,
                max_displacement=7.5,
                locked_borders=True
            ),
            tio.RandomNoise(p=0.1, std=(0, 0.25)), # Add noise
            tio.RandomBlur(p=0.1, std=(0.5, 1.5)), # Add blur
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        img_nii = nib.load(img_path)
        img_data = img_nii.get_fdata().astype(np.float32)

        # Load label
        lbl_path = self.label_files[idx]
        lbl_nii = nib.load(lbl_path)
        lbl_data = lbl_nii.get_fdata().astype(np.int32) # Labels should be integers

        # --- Add CT Windowing and Normalization ---
        # 1. Clip (Windowing) to a soft-tissue window (e.g., -100 to 400 HU)
        img_data = np.clip(img_data, -100, 400)
        
        # 2. Normalize the windowed image to [0, 1]
        # Add epsilon to prevent division by zero if max == min
        min_val = np.min(img_data)
        max_val = np.max(img_data)
        if max_val - min_val > 1e-6:
            img_data = (img_data - min_val) / (max_val - min_val)
        else:
            img_data = img_data - min_val # All values are the same, just set to 0

        # Resize and Pad
        img_resized = self.resize_and_pad(img_data.astype(np.float32), self.target_size)
        lbl_resized = self.resize_and_pad(lbl_data, self.target_size, is_label=True)

        # Add channel dimension (C, D, H, W)
        img_tensor = torch.from_numpy(img_resized).unsqueeze(0)
        lbl_tensor = torch.from_numpy(lbl_resized).long() 

        lbl_tensor_for_tio = lbl_tensor.unsqueeze(0)

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=img_tensor),
            label=tio.LabelMap(tensor=lbl_tensor_for_tio)
        )
        
        # This check now works, because self.transform was defined in __init__
        if self.augment:
            subject = self.transform(subject)
        
        # Return the augmented (or non-augmented) tensors
        return subject.image.data, subject.label.data.squeeze(0)

    def resize_and_pad(self, data, target_size, is_label=False):
        """
        Resizes (by padding or cropping) a 3D volume to a target size.
        """
        current_size = data.shape
        target_size = tuple(target_size) # (D, H, W)

        # Calculate padding
        pad_d = max(0, target_size[0] - current_size[0])
        pad_h = max(0, target_size[1] - current_size[1])
        pad_w = max(0, target_size[2] - current_size[2])

        pad_top = pad_d // 2
        pad_bottom = pad_d - pad_top
        pad_left = pad_h // 2
        pad_right = pad_h - pad_left
        pad_front = pad_w // 2
        pad_back = pad_w - pad_front

        padding = ((pad_top, pad_bottom), (pad_left, pad_right), (pad_front, pad_back))
        
        # Pad or crop
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            # We need to pad
            data_padded = np.pad(data, padding, mode='constant', constant_values=(0 if not is_label else 0))
        else:
            # No padding needed, but might need cropping
            data_padded = data
            
        # Cropping logic (if data is larger than target)
        crop_d = max(0, current_size[0] - target_size[0])
        crop_h = max(0, current_size[1] - target_size[1])
        crop_w = max(0, current_size[2] - target_size[2])

        start_d = crop_d // 2
        start_h = crop_h // 2
        start_w = crop_w // 2

        data_cropped = data_padded[
            start_d : start_d + target_size[0],
            start_h : start_h + target_size[1],
            start_w : start_w + target_size[2]
        ]
        
        # Final check for exact size (handles off-by-one in padding/cropping)
        if data_cropped.shape != target_size:
            final_data = np.zeros(target_size, dtype=data.dtype)
            min_d = min(target_size[0], data_cropped.shape[0])
            min_h = min(target_size[1], data_cropped.shape[1])
            min_w = min(target_size[2], data_cropped.shape[2])
            final_data[:min_d, :min_h, :min_w] = data_cropped[:min_d, :min_h, :min_w]
            return final_data
        
        return data_cropped
    
class LiverDataset_dist(Dataset):
    """
    Made by AI
    Custom PyTorch Dataset for the 3D Liver Segmentation data.
    """
    def __init__(self, image_dir, label_dir, target_size=INPUT_SHAPE):
        print(image_dir)
        print(label_dir)
        # --- THIS IS THE CORRECTED PART (looking for .nii) ---
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "imagesTr","*.nii")))
        self.label_paths = sorted(glob.glob(os.path.join(label_dir, "labelsTr" , "*.nii")))      
        self.target_size = target_size # (D, H, W)
        
        # Ensure we have matched pairs
        assert len(self.image_paths) > 0, f"No images found in {image_dir}"
        assert len(self.label_paths) > 0, f"No labels found in {label_dir}"
        assert len(self.image_paths) == len(self.label_paths), \
            f"Found {len(self.image_paths)} images but {len(self.label_paths)} labels."
        
        print(f"Found {len(self.image_paths)} image/label pairs.")

    def __len__(self):
        return len(self.image_paths)

    def normalize(self, data):
        # Normalize pixel values to [0, 1]
        data = data - torch.min(data)
        data = data / torch.max(data)
        return data
    def compute_distance_map(self, mask_np):
        """
        Computes the distance map for the boundary.
        mask_np: (D, H, W) numpy array
        """
        # We need the boundary, not the full mask.
        # This finds the boundary by checking for differences
        # with a shifted version of itself.
        boundary_np = np.logical_xor(
            mask_np, 
            ndimage.binary_erosion(mask_np)
        )
        
        if np.sum(boundary_np) == 0:
            # Handle empty mask, return all zeros
            return np.zeros_like(mask_np, dtype=np.float32)

        # Compute the distance transform.
        # This creates a map where each pixel's value
        # is its distance to the nearest boundary pixel.
        dist_map = ndimage.distance_transform_edt(~boundary_np)
        
        return dist_map.astype(np.float32)

    def __getitem__(self, idx):
        # 1. Load NIfTI files
        img_nii = nib.load(self.image_paths[idx])
        lbl_nii = nib.load(self.label_paths[idx])
        
        # 2. Get data as numpy array
        img_np = img_nii.get_fdata().astype(np.float32)
        lbl_np = lbl_nii.get_fdata().astype(np.int64)

        # 3. Permute to (D, H, W) for processing
        img_np = np.transpose(img_np, (2, 1, 0))
        lbl_np = np.transpose(lbl_np, (2, 1, 0))

        # 4. --- NEW --- Calculate distance map on numpy array
        # This map is the phi_G(q) from the paper
        dist_map_np = self.compute_distance_map(lbl_np)
        
        # 5. Convert to tensor (add channel dim)
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)
        lbl_tensor = torch.from_numpy(lbl_np).unsqueeze(0)
        dist_map_tensor = torch.from_numpy(dist_map_np).unsqueeze(0)

        # 6. Resize (trilinear for img/dist, nearest for label)
        img_resized = F.interpolate(img_tensor.unsqueeze(0), 
                                    size=self.target_size, 
                                    mode='trilinear', 
                                    align_corners=False).squeeze(0)
        
        dist_map_resized = F.interpolate(dist_map_tensor.unsqueeze(0),
                                         size=self.target_size,
                                         mode='trilinear',
                                         align_corners=False).squeeze(0)
        
        lbl_resized = F.interpolate(lbl_tensor.float().unsqueeze(0), 
                                    size=self.target_size, 
                                    mode='nearest').squeeze(0).long()

        # 7. Normalize image
        img_resized = self.normalize(img_resized)

        # 8. Squeeze the channel dim from the label
        lbl_resized = lbl_resized.squeeze(0) 

        # Return all three
        return img_resized, lbl_resized, dist_map_resized
    

class LiverUnlabeledDataset_aug(Dataset):
    """
    Dataset for Unlabeled 3D Liver Scans (e.g., imagesUnlabelledTr)
    
    Includes:
    - 3D Augmentations (via torchio)
    - CT Windowing (Clipping)
    - [0, 1] Normalization
    """
    def __init__(self, image_dir, subfolder, target_size=(128, 128, 128), augment=False):
        """
        Args:
            image_dir (Path): Path to the root data directory.
            subfolder (str): The subfolder containing unlabeled images.
            target_size (tuple): The desired (D, H, W) to resize/pad to.
            augment (bool): Whether to apply 3D augmentations.
        """
        self.image_dir = image_dir / subfolder
        self.target_size = target_size
        self.augment = augment

        self.image_files = sorted([f for f in self.image_dir.glob("*.nii")])

        if len(self.image_files) == 0:
            raise FileNotFoundError(f"No .nii files found in {self.image_dir}")
            
        print(f"Found {len(self.image_files)} unlabeled images.")

        # --- FIX: Define Augmentation Pipeline REGARDLESS of self.augment ---
        # The self.augment flag will be checked in __getitem__
        self.transform = tio.Compose([
            tio.RandomFlip(axes=('LR', 'AP', 'IS')),
            tio.RandomAffine(
                scales=(0.9, 1.2),
                degrees=15,
                isotropic=True,
                image_interpolation='linear'
            ),
            tio.RandomElasticDeformation(
                num_control_points=7,
                max_displacement=7.5,
                locked_borders=True
            ),
            tio.RandomNoise(p=0.25, std=(0, 0.25)), # More noise
            tio.RandomBlur(p=0.25, std=(0.5, 1.5)), # More blur
        ])
        # --- END FIX ---

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        img_nii = nib.load(img_path)
        img_data = img_nii.get_fdata().astype(np.float32)

        # --- Add CT Windowing and Normalization ---
        # 1. Clip (Windowing) to a soft-tissue window (e.g., -100 to 400 HU)
        img_data = np.clip(img_data, -100, 400)
        
        # 2. Normalize the windowed image to [0, 1]
        # Add epsilon to prevent division by zero if max == min
        min_val = np.min(img_data)
        max_val = np.max(img_data)
        if max_val - min_val > 1e-6:
            img_data = (img_data - min_val) / (max_val - min_val)
        else:
            img_data = img_data - min_val # All values are the same, just set to 0
        # --- END ---

        # Resize and Pad
        img_resized = self.resize_and_pad(img_data.astype(np.float32), self.target_size)

        # Add channel dimension
        img_tensor = torch.from_numpy(img_resized).unsqueeze(0) # (1, D, H, W)

        # Apply Augmentation (if enabled)
        # This check now works, because self.transform was defined in __init__
        if self.augment:
            # Torchio transforms expect (C, D, H, W)
            img_tensor = self.transform(img_tensor)

        # The loader expects a tuple, so we return (image,)
        return (img_tensor,)

    def resize_and_pad(self, data, target_size, is_label=False):
        """
        Resizes (by padding or cropping) a 3D volume to a target size.
        """
        # (Re-using the same function as above)
        current_size = data.shape
        target_size = tuple(target_size) # (D, H, W)

        pad_d = max(0, target_size[0] - current_size[0])
        pad_h = max(0, target_size[1] - current_size[1])
        pad_w = max(0, target_size[2] - current_size[2])

        pad_top = pad_d // 2
        pad_bottom = pad_d - pad_top
        pad_left = pad_h // 2
        pad_right = pad_h - pad_left
        pad_front = pad_w // 2
        pad_back = pad_w - pad_front

        padding = ((pad_top, pad_bottom), (pad_left, pad_right), (pad_front, pad_back))
        
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            data_padded = np.pad(data, padding, mode='constant', constant_values=(0 if not is_label else 0))
        else:
            data_padded = data
            
        crop_d = max(0, current_size[0] - target_size[0])
        crop_h = max(0, current_size[1] - target_size[1])
        crop_w = max(0, current_size[2] - target_size[2])

        start_d = crop_d // 2
        start_h = crop_h // 2
        start_w = crop_w // 2

        data_cropped = data_padded[
            start_d : start_d + target_size[0],
            start_h : start_h + target_size[1],
            start_w : start_w + target_size[2]
        ]
        
        if data_cropped.shape != target_size:
            final_data = np.zeros(target_size, dtype=data.dtype)
            min_d = min(target_size[0], data_cropped.shape[0])
            min_h = min(target_size[1], data_cropped.shape[1])
            min_w = min(target_size[2], data_cropped.shape[2])
            final_data[:min_d, :min_h, :min_w] = data_cropped[:min_d, :min_h, :min_w]
            return final_data
        
        return data_cropped
    

class LiverDataset(Dataset):
    """
    Custom PyTorch Dataset for the 3D Liver Segmentation data.
    
    This version includes:
    - Your glob.glob file loading
    - Your permute(2, 1, 0) logic
    - CT Windowing (clip -100 to 400) and [0, 1] Normalization
    - Padding / Cropping to target_size (NO interpolation)
    - NO augmentations
    """
    def __init__(self, image_dir, label_dir, target_size=INPUT_SHAPE):
        print(image_dir)
        print(label_dir)
        # Use your glob.glob logic to find .nii AND .nii.gz files
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "imagesTr", "*.nii*")))
        self.label_paths = sorted(glob.glob(os.path.join(label_dir, "labelsTr", "*.nii*")))      
        self.target_size = target_size # (D, H, W)
        
        # Ensure we have matched pairs
        assert len(self.image_paths) > 0, f"No images found in {os.path.join(image_dir, 'imagesTr')}"
        assert len(self.label_paths) > 0, f"No labels found in {os.path.join(label_dir, 'labelsTr')}"
        assert len(self.image_paths) == len(self.label_paths), \
            f"Found {len(self.image_paths)} images but {len(self.label_paths)} labels."
        
        print(f"Found {len(self.image_paths)} image/label pairs.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. Load NIfTI files
        img_nii = nib.load(self.image_paths[idx])
        lbl_nii = nib.load(self.label_paths[idx])
        
        # 2. Get data as numpy array
        img_data = img_nii.get_fdata().astype(np.float32)
        lbl_data = lbl_nii.get_fdata().astype(np.int32)

        # 3. Apply your permute logic (2, 1, 0) using numpy.transpose
        img_data = np.transpose(img_data, (2, 1, 0))
        lbl_data = np.transpose(lbl_data, (2, 1, 0))

        # 4. Apply CT Windowing and Normalization (on numpy array)
        # 4a. Clip (Windowing) to a soft-tissue window
        img_data = np.clip(img_data, -100, 400)
        
        # 4b. Normalize the windowed image to [0, 1]
        min_val = np.min(img_data)
        max_val = np.max(img_data)
        if max_val - min_val > 1e-6:
            img_data = (img_data - min_val) / (max_val - min_val)
        else:
            img_data = img_data - min_val # All values are the same, just set to 0

        # 5. Resize and Pad (cropping/padding)
        img_resized = self.resize_and_pad(img_data.astype(np.float32), self.target_size)
        lbl_resized = self.resize_and_pad(lbl_data, self.target_size, is_label=True)

        # 6. Convert to tensor and add channel dim
        img_tensor = torch.from_numpy(img_resized).unsqueeze(0) # Shape: [1, D, H, W]
        lbl_tensor = torch.from_numpy(lbl_resized).long()      # Shape: [D, H, W]

        return img_tensor, lbl_tensor

    def resize_and_pad(self, data, target_size, is_label=False):
        """
        Resizes (by padding or cropping) a 3D volume to a target size.
        """
        current_size = data.shape
        target_size = tuple(target_size) # (D, H, W)

        # Calculate padding
        pad_d = max(0, target_size[0] - current_size[0])
        pad_h = max(0, target_size[1] - current_size[1])
        pad_w = max(0, target_size[2] - current_size[2])

        pad_top = pad_d // 2
        pad_bottom = pad_d - pad_top
        pad_left = pad_h // 2
        pad_right = pad_h - pad_left
        pad_front = pad_w // 2
        pad_back = pad_w - pad_front

        padding = ((pad_top, pad_bottom), (pad_left, pad_right), (pad_front, pad_back))
        
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            data_padded = np.pad(data, padding, mode='constant', constant_values=(0 if not is_label else 0))
        else:
            data_padded = data
            
        # Cropping logic (if data is larger than target)
        crop_d = max(0, current_size[0] - target_size[0])
        crop_h = max(0, current_size[1] - target_size[1])
        crop_w = max(0, current_size[2] - target_size[2])

        start_d = crop_d // 2
        start_h = crop_h // 2
        start_w = crop_w // 2

        data_cropped = data_padded[
            start_d : start_d + target_size[0],
            start_h : start_h + target_size[1],
            start_w : start_w + target_size[2]
        ]
        
        if data_cropped.shape != target_size:
            final_data = np.zeros(target_size, dtype=data.dtype)
            min_d = min(target_size[0], data_cropped.shape[0])
            min_h = min(target_size[1], data_cropped.shape[1])
            min_w = min(target_size[2], data_cropped.shape[2])
            final_data[:min_d, :min_h, :min_w] = data_cropped[:min_d, :min_h, :min_w]
            return final_data
        
        return data_cropped

    
class LiverUnlabeledDataset(Dataset):
    """
    Custom PyTorch Dataset for 3D Liver UNLABELED images.
    
    This version includes:
    - Your glob.glob file loading
    - Your permute(2, 1, 0) logic
    - CT Windowing (clip -100 to 400) and [0, 1] Normalization
    - Padding / Cropping to target_size (NO interpolation)
    - NO augmentations
    """
    def __init__(self, image_dir, target_size=INPUT_SHAPE, subfolder="imagesUnlabelledTr"):
        # Assumes unlabeled images are in a folder like 'imagesUnlabeledTr'
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, subfolder, "*.nii*")))
        self.target_size = target_size # (D, H, W)
        
        assert len(self.image_paths) > 0, f"No unlabeled images found in {os.path.join(image_dir, subfolder)}"
        print(f"Found {len(self.image_paths)} unlabeled images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. Load NIfTI file
        img_nii = nib.load(self.image_paths[idx])
        
        # 2. Get data as numpy array
        img_data = img_nii.get_fdata().astype(np.float32)

        # 3. Apply your permute logic (2, 1, 0)
        img_data = np.transpose(img_data, (2, 1, 0))

        # 4. Apply CT Windowing and Normalization
        img_data = np.clip(img_data, -100, 400)
        min_val = np.min(img_data)
        max_val = np.max(img_data)
        if max_val - min_val > 1e-6:
            img_data = (img_data - min_val) / (max_val - min_val)
        else:
            img_data = img_data - min_val

        # 5. Resize and Pad (cropping/padding)
        img_resized = self.resize_and_pad(img_data.astype(np.float32), self.target_size)

        # 6. Convert to tensor and add channel dim
        img_tensor = torch.from_numpy(img_resized).unsqueeze(0) # Shape: [1, D, H, W]

        # 7. Return as a 1-item tuple
        return (img_tensor,)

    def resize_and_pad(self, data, target_size, is_label=False):
        """
        Resizes (by padding or cropping) a 3D volume to a target size.
        """
        current_size = data.shape
        target_size = tuple(target_size) # (D, H, W)

        # Calculate padding
        pad_d = max(0, target_size[0] - current_size[0])
        pad_h = max(0, target_size[1] - current_size[1])
        pad_w = max(0, target_size[2] - current_size[2])

        pad_top = pad_d // 2
        pad_bottom = pad_d - pad_top
        pad_left = pad_h // 2
        pad_right = pad_h - pad_left
        pad_front = pad_w // 2
        pad_back = pad_w - pad_front

        padding = ((pad_top, pad_bottom), (pad_left, pad_right), (pad_front, pad_back))
        
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            data_padded = np.pad(data, padding, mode='constant', constant_values=(0 if not is_label else 0))
        else:
            data_padded = data
            
        # Cropping logic (if data is larger than target)
        crop_d = max(0, current_size[0] - target_size[0])
        crop_h = max(0, current_size[1] - target_size[1])
        crop_w = max(0, current_size[2] - target_size[2])

        start_d = crop_d // 2
        start_h = crop_h // 2
        start_w = crop_w // 2

        data_cropped = data_padded[
            start_d : start_d + target_size[0],
            start_h : start_h + target_size[1],
            start_w : start_w + target_size[2]
        ]
        
        if data_cropped.shape != target_size:
            final_data = np.zeros(target_size, dtype=data.dtype)
            min_d = min(target_size[0], data_cropped.shape[0])
            min_h = min(target_size[1], data_cropped.shape[1])
            min_w = min(target_size[2], data_cropped.shape[2])
            final_data[:min_d, :min_h, :min_w] = data_cropped[:min_d, :min_h, :min_w]
            return final_data
        
        return data_cropped