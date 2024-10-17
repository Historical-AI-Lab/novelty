#!/bin/bash
#SBATCH --mem=62g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4   # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA40x4      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bbiq-delta-gpu
#SBATCH --time=24:00:00                    # Job run time (hh:mm:ss)
#SBATCH --job-name=econmod1904                # Name of batch job"
#SBATCH --error="econmod1904_%j.err"
#SBATCH --output="econmod1904_%j.out"
##SBATCH --mail-user=tunder@illinois.edu   # Send email notifications
##SBATCH --mail-type="BEGIN,END"             # Type of email notifications to send
### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none     # <- or closest

echo "job is starting on `hostname`"
source py311gpu/bin/activate
python3.11 Model12YearsOfEcon.py ../metadata/all-econ-S2metaWyearcol.tsv 1904 1915 all-econ-chunks
