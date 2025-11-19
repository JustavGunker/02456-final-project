import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
from skimage.util import random_noise
from pathlib import Path
import itertools # Needed for generating 3D partition indices

BLACKHOLE_PATH = os.environ.get('BLACKHOLE', '.')
BASE_DATA_DIR = os.path.join(BLACKHOLE_PATH, 'deep_learning_214776', 'extracted_datasets', 'datasets_processed_latest')

NUM_COLUMNS = 38 
VOLUME_SIZE = 256
PATCH_SIZE = 128
SLICES_PER_AXIS = VOLUME_SIZE // PATCH_SIZE # 256 / 64 = 4

TARGET_DEPTH = PATCH_SIZE
TARGET_HEIGHT = PATCH_SIZE
TARGET_WIDTH = PATCH_SIZE

# --- The master list of all 64 unique partition indices (pd, ph, pw) ---
PARTITION_INDICES = list(itertools.product(range(SLICES_PER_AXIS), repeat=3))
# -------------------------------------

class VolumetricPatchDataset(Dataset):
    """
    PyTorch Dataset for memory-efficient loading and on-the-fly 3D partitioning.
    Creates 64 patches of size (64, 64, 64) from each original 256^3 volume.
    """
    def __init__(self, base_dir=BASE_DATA_DIR, num_columns=NUM_COLUMNS, augment=True, is_labeled=True):
        self.base_dir = Path(base_dir)
        self.augment = augment
        self.is_labeled = is_labeled
        
        self.sample_paths = self._create_sample_list(num_columns)
        
        # Check if the path is valid right after initialization
        if not self.base_dir.exists():
            print(f"ERROR: Data path not found: {self.base_dir}. Ensure BLACKHOLE is set and data is moved.")
        
        print(f"Dataset initialized with {len(self.sample_paths)} total patches.")

    def _create_sample_list(self, num_columns):
        """Generates a list of (column_number, half_type, partition_index_tuple)."""
        paths = []
        for i in range(1, num_columns + 1):
            column_dir = self.base_dir / f'Column_{i}'
            
            for p_idx_tuple in PARTITION_INDICES: # Iterates 64 times
                # Check for TOP half presence
                if (column_dir / 'B' / 'top.mat').exists():
                     paths.append((i, 'top', p_idx_tuple))
                # Check for BOTTOM half presence
                if (column_dir / 'B' / 'bottom.mat').exists():
                     paths.append((i, 'bottom', p_idx_tuple))
        return paths

    def __len__(self):
        """Returns the total number of patches (approx. 4864)."""
        return len(self.sample_paths)

    def __getitem__(self, index):
        """Loads and processes one patch."""
        column_num, half_type, p_idx_tuple = self.sample_paths[index]
        
        # Load sample from disk
        X_patch, Y_patch = self._load_patch(column_num, half_type, p_idx_tuple)
        
        # Apply augmentation (if required)
        if self.augment:
            X_patch, Y_patch = self._augment_sample(X_patch, Y_patch)

        # Convert to PyTorch tensors and ensure correct dtype/device handling (float32 for model inputs)
        X_tensor = torch.from_numpy(X_patch).float()
        
        if self.is_labeled:
            Y_tensor = torch.from_numpy(Y_patch).long()
            return X_tensor, Y_tensor
        else:
            return X_tensor
    
    def _load_patch(self, column_num, half_type, p_idx_tuple):
        """Loads a full volume, slices it into the 64x64x64 patch, and performs normalization."""
        column_dir = self.base_dir / f'Column_{column_num}'
        
        # Get the partition indices (pd, ph, pw)
        pd, ph, pw = p_idx_tuple

        if half_type == 'top':
            x_filepath = column_dir / 'B' / 'top.mat'
            y_filepath = column_dir / 'gt_top.mat'
            x_var_name = 'top'
            y_var_name = 'gt_top'
        else: # 'bottom'
            x_filepath = column_dir / 'B' / 'bottom.mat'
            y_filepath = column_dir / 'gt_bottom.mat'
            x_var_name = 'bottom'
            y_var_name = 'gt_bottom'

        try:
            X_full = np.squeeze(loadmat(x_filepath)[x_var_name])
            Y_full = np.squeeze(loadmat(y_filepath)[y_var_name])
            
            # --- APPLY 3D PARTITION SLICING ---
            d_start, d_end = pd * PATCH_SIZE, (pd + 1) * PATCH_SIZE
            h_start, h_end = ph * PATCH_SIZE, (ph + 1) * PATCH_SIZE
            w_start, w_end = pw * PATCH_SIZE, (pw + 1) * PATCH_SIZE
            
            X_patch = X_full[d_start:d_end, h_start:h_end, w_start:w_end]
            Y_patch = Y_full[d_start:d_end, h_start:h_end, w_start:w_end]

            # Normalization
            X_patch = X_patch.astype(np.float32) / 255.0

            # Critical fixes for PyTorch memory
            X_patch = X_patch.copy()
            Y_patch = Y_patch.copy()

            # Add Channel dimension (C) to the front for PyTorch: (D, H, W) -> (C, D, H, W)
            X_patch = np.expand_dims(X_patch, axis=0)
            Y_patch = np.expand_dims(Y_patch, axis=0)
            
            if self.is_labeled:
                 Y_patch = Y_patch.astype(np.int64)

            return X_patch, Y_patch

        except Exception as e:
            raise RuntimeError(f"Failed to load patch {column_num}/{half_type}/{p_idx_tuple}: {e}")

    def _augment_sample(self, X_patch, Y_patch):
        """Applies stable 3D augmentation to a single sample."""
        # X_patch shape: (C, D, H, W)
        
        # NumPy operations on (D, H, W) axes (axes 1, 2, 3 in the tensor)
        X = X_patch.squeeze(0)
        Y = Y_patch.squeeze(0)
        
        # Random Flips (still applied in 3D)
        if np.random.rand() > 0.5:
            X = np.flip(X, axis=0); Y = np.flip(Y, axis=0);
        if np.random.rand() > 0.5:
            X = np.flip(X, axis=1); Y = np.flip(Y, axis=1);
        if np.random.rand() > 0.5:
            X = np.flip(X, axis=2); Y = np.flip(Y, axis=2);

        # Add Gaussian Noise
        X = random_noise(X, mode='gaussian', var=0.005) 

        # Final memory copy fix
        X_augmented = np.expand_dims(X.copy(), axis=0)
        Y_augmented = np.expand_dims(Y.copy(), axis=0)
            
        return X_augmented, Y_augmented