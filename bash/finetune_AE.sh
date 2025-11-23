#!/bin/sh
### --- Job Name ---
#BSUB -J pretain_AE_job

### --- Log files ---
#BSUB -o /zhome/d2/4/167803/Desktop/Deep_project/02456-final-project/logs/finetune_AE_job_%J.out
#BSUB -e /zhome/d2/4/167803/Desktop/Deep_project/02456-final-project/logs/finetune_AE_job_%J.err
### --- Resource Requests ---
#BSUB -q gpua100             # Request GPU queue
#BSUB -gpu "num=1:mode=exclusive_process" # Request 1 GPU, all to myself
#BSUB -n 4                    # Request 4 CPU cores
#BSUB -R "rusage[mem=12GB]"   # Request 16GB memory
#BSUB -W 04:00                # 4 hour runtime limit

### --- Setup Environment ---
echo "Loading CUDA module..."
module load cuda/11.6         # Load CUDA module

echo "Loading Conda..."
# This is the full, correct path to your environment
source /zhome/d2/4/167803/miniforge3/bin/activate MBML
echo "Environment 'MBML' activated."

### --- Run Script ---
echo "Starting Python script..."
python /zhome/d2/4/167803/Desktop/Deep_project/02456-final-project/Model/finetune_AE.py

echo "Job Finished."