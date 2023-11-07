#!/bin/bash
#SBATCH --mem=16g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1   # <- match to OMP_NUM_THREADS
#SBATCH --partition=cpu      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bbiq-delta-cpu
#SBATCH --time=24:00:00                    # Job run time (hh:mm:ss)
#SBATCH --job-name=ch1000                # Name of batch job"
#SBATCH --error="ch1000_%j.err"
#SBATCH --output="ch1000_%j.out"
##SBATCH --mail-user=tunder@illinois.edu   # Send email notifications
##SBATCH --mail-type="BEGIN,END"             # Type of email notifications to send
### GPU options ###
###SBATCH --gpus-per-node=1
###SBATCH --gpu-bind=none     # <- or closest

echo "job is starting on `hostname`"
source embedenv/bin/activate
srun python3 chunk_texts.py 1000
