#!/bin/bash
#SBATCH --mem=62g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1   # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA40x4      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bbiq-delta-gpu
#SBATCH --time=24:00:00                    # Job run time (hh:mm:ss)
#SBATCH --job-name=ficmod1894                # Name of batch job"
#SBATCH --error="ficmod1894_%j.err"
#SBATCH --output="ficmod1894_%j.out"
##SBATCH --mail-user=tunder@illinois.edu   # Send email notifications
##SBATCH --mail-type="BEGIN,END"             # Type of email notifications to send
### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none     # <- or closest

echo "job is starting on `hostname`"
source py311gpu/bin/activate
python3.11 Model12YearsOfFiction.py ../fiction/capped_fiction_metadata.tsv 1894 1905 ../fiction/fiction_chunks
