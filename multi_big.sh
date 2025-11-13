#!/bin/sh
### --- Job Name ---
#BSUB -J My_MultiTask_Training

### --- Log files ---
#BSUB -o /zhome/d2/4/167803/Desktop/Deep_project/02456-final-project/logs/job_%J.out
#BSUB -e /zhome/d2/4/167803/Desktop/Deep_project/02456-final-project/logs/job_%J.err

### --- Resource Requests ---
#BSUB -q gpua40               # Request the A100 queue
#BSUB -gpu "num=1:mode=exclusive_process" # Request 1 GPU, all to myself
#BSUB -n 4                    # Request 4 CPU cores for data loading
#BSUB -R "rusage[mem=32GB]"   # Request 32GB of main RAM (not GPU RAM)
#BSUB -W 04:00                # Set walltime limit to 4 hours (hh:mm)

### --- Setup Environment ---
echo "Loading Conda..."
# This is the full, correct path to your environment
source /zhome/d2/4/167803/miniforge3/bin/activate MBML
echo "Environment 'MBML' activated."

### --- Run Script ---
echo "Starting Python script..."
python /zhome/d2/4/167803/Desktop/Deep_project/02456-final-project/Model/VAE.py

echo "Job Finished."
