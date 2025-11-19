import os
import sys
from scipy.io import loadmat
import numpy as np
from pathlib import Path

# --- Configuration ---
# NOTE: Uses the same path construction as your data loader to ensure accuracy.
BLACKHOLE_PATH = os.environ.get('BLACKHOLE', '.')
BASE_DATA_DIR = os.path.join(
    BLACKHOLE_PATH, 
    'deep_learning_214776', 
    'extracted_datasets', 
    'datasets_processed_latest'
)

# We will check the first column and the 'top' half.
COLUMN_NAME = 'Column_1'
FILE_NAME = 'top.mat'
VARIABLE_NAME = 'top' # The variable name inside the .mat file

# --- Main Logic ---

def check_volume_shape():
    """Loads a single .mat file and prints its raw shape."""
    
    data_path = Path(BASE_DATA_DIR)
    target_filepath = data_path / COLUMN_NAME / 'B' / FILE_NAME
    
    print(f"\n--- Checking Raw Volume Shape ---")
    print(f"Loading from: {target_filepath}")

    if not target_filepath.exists():
        print(f"\n❌ ERROR: File not found at {target_filepath}")
        print("Please ensure the BLACKHOLE variable is correctly set and the data was moved correctly.")
        return

    try:
        # Load the .mat file
        mat_contents = loadmat(str(target_filepath))
        
        # Extract the specific variable ('top')
        if VARIABLE_NAME not in mat_contents:
            print(f"❌ ERROR: Variable '{VARIABLE_NAME}' not found inside the .mat file.")
            return

        # Extract the array and squeeze any single-dimensional axes
        volume_array = np.squeeze(mat_contents[VARIABLE_NAME])
        
        # Print the final shape
        print("\n✅ Success!")
        print(f"Volume Array Shape (D, H, W): {volume_array.shape}")
        
        # Confirmation based on the expected 256^3 size
        if volume_array.shape == (256, 256, 256):
            print("Confirmation: This matches the expected 256x256x256 original volume size.")
        else:
            print("Warning: Shape does not match the expected (256, 256, 256).")

    except Exception as e:
        print(f"\n❌ FATAL ERROR during loading: {e}")

if __name__ == "__main__":
    check_volume_shape()