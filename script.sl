#!/bin/bash

# Slurm submission script, serial job
# CRIHAN v 1.00 - Jan 2017
# support@criann.fr

# Time limit for the calculation (48:00:00 max)
#SBATCH --time 48:00:00

# Memory to use (here 50Go)
#SBATCH --mem 150000

# Type of gpu to use, either gpu_all, gpu_k80, gpu_p100 or gpu_v100
#SBATCH --partition gpu_v100

# Number of gpu to use
#SBATCH --gres gpu:4

# Number of tasks 
#SBATCH --ntasks-per-node=4

# Number of node to use
#SBATCH --nodes 2

# Number of cpu to use
#SBATCH --cpus-per-task=6

# Were to write the logs
#SBATCH --error slurm/%J.err
#SBATCH --output slurm/%J.out

module load python3-DL
export PYTHONUSERBASE=/home/2021012/sruan01/riles/env
pip install pytorch-lightning --user
pip install nibabel --user
pip install wandb --user
pip install omegaconf --user
pip install shortuuid --user

# Start the calculation (safer to use srun)
srun python3 $1
