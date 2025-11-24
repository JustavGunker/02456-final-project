import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# --- Configuration ---
# Set PROJECT_ROOT to the main directory that CONTAINS the 'func' folder, 
# allowing imports like 'from func.dataloaders' to work.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import the shared Dataset class
try:
    # --- CORRECTED IMPORT PATH based on user's structure ---
    from func.dataloaders import VolumetricPatchDataset
except ImportError as e:
    print(f"ERROR: Could not import VolumetricPatchDataset. Details: {e}")
    print("Please ensure the VolumetricPatchDataset class is defined in func/dataloaders.py.")
    sys.exit(1)

# Configuration must match the constant in volumetric_dataset.py
NUM_CLASSES = 4 
# ---------------------

def visualize_single_slice():
    """
    Instantiates the dataset and visualizes the central 2D slice of the first 3D patch.
    Saves the image and the corresponding label ground truth.
    """
    print("\n--- Starting Standalone Visualization Script ---")
    
    try:
        # Instantiate the dataset (Augmentation is off for clean visualization)
        dataset = VolumetricPatchDataset(augment=False, is_labeled=True)
        
        if len(dataset) == 0:
            print("Error: Dataset is empty. Check data paths in the dataset class configuration.")
            return

        # Fetch the first sample (X_tensor, Y_tensor)
        X_patch_tensor, Y_patch_tensor = dataset[0]

        # Convert back to NumPy for slicing/plotting. Shape: (C, D, H, W)
        X_patch = X_patch_tensor.numpy()
        Y_patch = Y_patch_tensor.numpy()
        
        # 1. Extract the central slice along the D axis (D=64, so index 32)
        middle_slice_index = X_patch.shape[1] // 2 
        
        # Extract the 2D slices. Shape: (256, 256)
        image_slice = X_patch[0, middle_slice_index, :, :]
        label_slice = Y_patch[0, middle_slice_index, :, :]
        
        # 2. Print verification info
        print(f"Total Patches Available: {len(dataset)}")
        print(f"Patch Input Shape (C, D, H, W): {X_patch.shape}")
        print(f"Extracted Slice Shape (H, W): {image_slice.shape}")
        print(f"Min/Max Intensity (Normalized): {np.min(image_slice):.4f} / {np.max(image_slice):.4f}")
        
        # 3. Save the slices as PNG images
        
        # Input Image Slice (Grayscale)
        input_filename = "visualize_patch_input.png"
        plt.imsave(input_filename, image_slice, cmap='gray')
        print(f"✅ Input slice saved successfully to: {input_filename}")

        # Label Slice (Categorical color map)
        label_filename = "visualize_patch_label.png"
        # Use 'viridis' or 'nipy_spectral' for distinct classes, ensuring limits match 0 to 3
        plt.imsave(label_filename, label_slice, cmap='viridis', vmin=0, vmax=NUM_CLASSES-1)
        print(f"✅ Label slice saved successfully to: {label_filename}")
        
    except Exception as e:
        print(f"\nFATAL ERROR during visualization: {e}")
        print("Hint: Ensure Matplotlib is installed (conda install matplotlib) and data path is correct.")
        sys.exit(1)

if __name__ == "__main__":
    visualize_single_slice()