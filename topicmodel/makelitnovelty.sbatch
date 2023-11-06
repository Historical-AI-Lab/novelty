#!/bin/bash
###############################################################################
##                                                                           ##
##                           Campus Cluster                                  ##
##                    Sample OpenMP Job Batch Script                         ##
##                                                                           ##
## SLURM Options (To view, run the following command below)                  ##
##                                                                           ##
##     man sbatch                                                            ##
##                                                                           ##
###############################################################################
#SBATCH --mem=32g                                                                            
#SBATCH --time=10:00:00                    # Job run time (hh:mm:ss)
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --ntasks-per-node=1               # Number of task (cores/ppn) per node
#SBATCH --job-name=makeavant             # Name of batch job
#SBATCH --partition=ichass              # Partition (queue)           
#SBATCH --output=openmp.o%j                # Name of batch job output file
#SBATCH --error=openmp.e%j                # Name of batch job error file
##SBATCH --mail-user=tunder@illinois.edu  # Send email notifications
##SBATCH --mail-type=BEGIN,END             # Type of email notifications to send
#                                                                            
###############################################################################

# Change to the directory from which the batch job was submitted
# Note: SLURM defaults to running jobs in the directory where
# they are submitted, no need for cd'ing to $SLURM_SUBMIT_DIR

#cd ${SLURM_SUBMIT_DIR}

# Add scripts to run here
bin/mallet import-file --input ../novelty/embeddingcode/litstudiesforLDA.txt --output ../novelty/topicmodel/litstudies.mallet --keep-sequence --token-regex '\p{L}[\p{L}\p{P}]*\p{L}|\p{L}'
