#!/bin/bash
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA40x4      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bbiq-delta-gpu
#SBATCH --time=02:00:00                    # Job run time (hh:mm:ss)
#SBATCH --job-name=chunktexts                # Name of batch job"
##SBATCH --mail-user=tunder@illinois.edu   # Send email notifications
##SBATCH --mail-type="BEGIN,END"             # Type of email notifications to send
### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none     # <- or closest

echo "job is starting on `hostname`"
srun python3 chunk_texts.py
