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
#SBATCH --time=36:00:00                    # Job run time (hh:mm:ss)
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --ntasks-per-node=20               # Number of task (cores/ppn) per node
#SBATCH --job-name=flatlit250                # Name of batch job
#SBATCH --partition=IllinoisComputes                 # Partition (queue)
#SBATCH --account=tunder-ic           
#SBATCH --output=lit2flat.o%j                # Name of batch job output file
##SBATCH --error=lit2flat.e%j                # Name of batch job error file
##SBATCH --mail-user=tunder@illinois.edu   # Send email notifications
##SBATCH --mail-type=BEGIN,END             # Type of email notifications to send
#                                                                            
###############################################################################

# Change to the directory from which the batch job was submitted
# Note: SLURM defaults to running jobs in the directory where
# they are submitted, no need for cd'ing to $SLURM_SUBMIT_DIR

#cd ${SLURM_SUBMIT_DIR}

# Add scripts to run here
bin/mallet train-topics --input ../novelty/topicmodel/LitstudiesFlat.mallet --num-threads 18 --num-topics 250 --num-iterations 2000 --num-top-words 80 --random-seed 99 --optimize-interval 20 --optimize-burn-in 100 --output-state ../novelty/topicmodel/flat250state.gz --output-doc-topics ../novelty/topicmodel/flat250doctopics.txt --output-topic-keys ../novelty/topicmodel/flat250keys.txt --show-topics-interval 200 --diagnostics-file ../novelty/topicmodel/flat250diagnostics.xml --inferencer-filename ../novelty/topicmodel/flat250inferencer
