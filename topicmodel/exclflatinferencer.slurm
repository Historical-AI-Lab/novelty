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
#                                                                            
#SBATCH --time=12:00:00                    # Job run time (hh:mm:ss)
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --ntasks-per-node=16               # Number of task (cores/ppn) per node
#SBATCH --job-name=goodinference                # Name of batch job
#SBATCH --partition=IllinoisComputes                # Partition (queue) 
#SBATCH --account=tunder-ic          
#SBATCH --output=inference.o%j                # Name of batch job output file
##SBATCH --error=inference.e%j                # Name of batch job error file
##SBATCH --mail-user=tunder@illinois.edu   # Send email notifications
##SBATCH --mail-type=BEGIN,END             # Type of email notifications to send
#                                                                            
###############################################################################

# Change to the directory from which the batch job was submitted
# Note: SLURM defaults to running jobs in the directory where
# they are submitted, no need for cd'ing to $SLURM_SUBMIT_DIR

#cd ${SLURM_SUBMIT_DIR}

# Add scripts to run here
bin/mallet infer-topics --inferencer ../novelty/topicmodel/flat250inferencer --input ../novelty/topicmodel/ExcludedFromFlat.mallet --num-iterations 120 --output-doc-topics ../novelty/topicmodel/excl_from_flat_doctopic_best.txt
