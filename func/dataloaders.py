import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
from skimage.util import random_noise
from pathlib import Path
import itertools # Needed for generating 3D partition indices
import tifffile

BLACKHOLE_PATH = os.environ.get('BLACKHOLE', '.')
BASE_DATA_DIR = os.path.join(BLACKHOLE_PATH, 'deep_learning_214776', 'extracted_datasets', 'datasets_processed_latest')

VOLUME_SIZE = 256
PATCH_SIZE = 128
SLICES_PER_AXIS = VOLUME_SIZE // PATCH_SIZE # 256 / 128 = 2
PARTITION_INDICES = list(itertools.product(range(SLICES_PER_AXIS), repeat=3))

class VolumetricPatchDataset(Dataset):
    def __init__(self, selected_columns, base_dir=BASE_DATA_DIR, augment=True, is_labeled=True):
        """
        Args:
            selected_columns (list): List of integers representing the Column IDs (e.g., [1, 2, 3])
            base_dir (str): Path to data
            augment (bool): Apply rotation/noise?
            is_labeled (bool): Return (X, Y) or just X?
        """
        self.base_dir = Path(base_dir)
        self.augment = augment
        self.is_labeled = is_labeled
        
        # Pass the specific list of columns to the sample creator
        self.sample_paths = self._create_sample_list(selected_columns)
        
        if len(self.sample_paths) == 0:
            print(f"WARNING: No files found for columns {selected_columns}. Check path: {self.base_dir}")

    def _create_sample_list(self, selected_columns):
        paths = []
        for col_idx in selected_columns:
            column_dir = self.base_dir / f'Column_{col_idx}'
            
            # Create patches for every partition index
            for p_idx_tuple in PARTITION_INDICES: 
                if (column_dir / 'B' / 'top.mat').exists():
                     paths.append((col_idx, 'top', p_idx_tuple))
                if (column_dir / 'B' / 'bottom.mat').exists():
                     paths.append((col_idx, 'bottom', p_idx_tuple))
        return paths

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, index):
        column_num, half_type, p_idx_tuple = self.sample_paths[index]
        
        try:
            X_patch, Y_patch = self._load_patch(column_num, half_type, p_idx_tuple)
            
            if self.augment:
                X_patch, Y_patch = self._augment_sample(X_patch, Y_patch)

            X_tensor = torch.from_numpy(X_patch).float()
            
            if self.is_labeled:
                Y_tensor = torch.from_numpy(Y_patch).long()
                return X_tensor, Y_tensor
            else:
                return X_tensor
        except Exception as e:
            print(f"Error loading {column_num} {half_type}: {e}")
            # Return a zero tensor to avoid crashing, or handle appropriately
            return torch.zeros((1, 128, 128, 128)), torch.zeros((128, 128, 128))

    def _load_patch(self, column_num, half_type, p_idx_tuple):
        column_dir = self.base_dir / f'Column_{column_num}'
        pd, ph, pw = p_idx_tuple # Partition indices

        x_filepath = column_dir / 'B' / f'{half_type}.mat'
        y_filepath = column_dir / f'gt_{half_type}.mat'
        
        # Load and Squeeze
        X_full = np.squeeze(loadmat(str(x_filepath))[half_type])
        Y_full = np.squeeze(loadmat(str(y_filepath))[f'gt_{half_type}'])
        
        # Calculate Slices
        d_start, d_end = pd * PATCH_SIZE, (pd + 1) * PATCH_SIZE
        h_start, h_end = ph * PATCH_SIZE, (ph + 1) * PATCH_SIZE
        w_start, w_end = pw * PATCH_SIZE, (pw + 1) * PATCH_SIZE
        
        # Slice
        X_patch = X_full[d_start:d_end, h_start:h_end, w_start:w_end]
        Y_patch = Y_full[d_start:d_end, h_start:h_end, w_start:w_end]

        # Normalize
        X_patch = X_patch.astype(np.float32) / 255.0
        
        # Add Channel Dim
        X_patch = np.expand_dims(X_patch.copy(), axis=0)
        Y_patch = Y_patch.copy() # Keep shape (D,H,W) for labels

        return X_patch, Y_patch

    def _augment_sample(self, X_patch, Y_patch):
        X = X_patch.squeeze(0)
        Y = Y_patch
        
        if np.random.rand() > 0.5:
            X = np.flip(X, axis=0); Y = np.flip(Y, axis=0)
        if np.random.rand() > 0.5:
            X = np.flip(X, axis=1); Y = np.flip(Y, axis=1)
        if np.random.rand() > 0.5:
            X = np.flip(X, axis=2); Y = np.flip(Y, axis=2)

        X = random_noise(X, mode='gaussian', var=0.005) 
        # intensity mod and rotation and constract

        X_augmented = np.expand_dims(X.copy(), axis=0)
        Y_augmented = Y.copy()
            
        return X_augmented, Y_augmented
