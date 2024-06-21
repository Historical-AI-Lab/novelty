#!/bin/bash
#SBATCH --mem=120g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1   # <- match to OMP_NUM_THREADS
#SBATCH --partition=IllinoisComputes      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=tunder-ic
#SBATCH --time=36:00:00                    # Job run time (hh:mm:ss)
#SBATCH --job-name=fic1950-54                # Name of batch job"
#SBATCH --error="fic1950-54_%j.err"
#SBATCH --output="fic1950-54_%j.out"
##SBATCH --mail-user=tunder@illinois.edu   # Send email notifications
##SBATCH --mail-type="END"             # Type of email notifications to send
### GPU options ###

echo "job is starting on `hostname`"
module load python/3.9.16
source ../embeddingcode/embedenv/bin/activate
python3.9 calc_precocity_kld.py -m ../fiction/capped_fiction_metadata.tsv -d fiction/doctopics -c ../fiction/no_chunk_exclusions.tsv -s 1950 -e 1955 -a ../fiction/author_exclusions_fiction.tsv -o ficresults
