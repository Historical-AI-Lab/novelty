#!/bin/bash
#SBATCH --mem=16g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1   # <- match to OMP_NUM_THREADS
#SBATCH --partition=ichass      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --time=24:00:00                    # Job run time (hh:mm:ss)
#SBATCH --job-name=try3_5k                # Name of batch job"
#SBATCH --error="try3_5k_%j.err"
#SBATCH --output="try3_5k_%j.out"
##SBATCH --mail-user=tunder@illinois.edu   # Send email notifications
##SBATCH --mail-type="BEGIN,END"             # Type of email notifications to send
### GPU options ###

echo "job is starting on `hostname`"
module load python/3.9.16
source embedenv/bin/activate
python3.9 chunk_and_embed.py 3000 jsonfile.jsonl metadata.tsv
