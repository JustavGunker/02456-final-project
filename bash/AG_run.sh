#!/bin/sh
### --- Job Name ---
#BSUB -J AG_job

### --- Log files ---
#BSUB -o /zhome/d2/4/167803/Desktop/Deep_project/02456-final-project/logs/AG_job_%J.out
#BSUB -e /zhome/d2/4/167803/Desktop/Deep_project/02456-final-project/logs/AG_err_%J.err

### --- Resource Requests ---
#BSUB -q gpul40s           # Request GPU queue
#BSUB -gpu "num=1:mode=exclusive_process" # Request 1 GPU, all to myself
#BSUB -n 4                    # Request 4 CPU cores
#BSUB -R "rusage[mem=8GB]"   # Request 16GB memory
#BSUB -W 04:00                # 4 hour runtime limit

### --- Setup Environment ---
echo "Loading CUDA module..."
module load cuda/11.6         # Load CUDA module

echo "Loading Conda..."

source /zhome/d2/4/167803/miniforge3/bin/activate MBML
echo "Environment 'MBML' activated."

echo "Starting Python script..."
python /zhome/d2/4/167803/Desktop/Deep_project/02456-final-project/Model/AG.py

echo "Job Finished."