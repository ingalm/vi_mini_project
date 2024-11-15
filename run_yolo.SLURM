#!/bin/sh
#SBATCH --account=ie-idi       # E.g. "ie-idi" if you belong to IDI
#SBATCH --job-name=VI-miniproject
#SBATCH --time=0-10:00:00         # format: D-HH:MM:SS

#SBATCH --partition=GPUQ          # Asking for a GPU
#SBATCH --gres=gpu:2             # Setting the number of GPUs to 1
#SBATCH --mem=16G                 # Asking for 16GB RAM
#SBATCH --nodes=1
#SBATCH --output=output.txt      # Specifying 'stdout'
#SBATCH --error=output.err        # Specifying 'stderr'

#SBATCH --mail-user=evaajo@stud.ntnu.no
#SBATCH --mail-type=ALL

module purge

# Load necessary modules
module load Anaconda3/2024.02-1
module load torchvision/0.16.0-foss-2023a-CUDA-12.1.1

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

# Run your Python program
~/.conda/envs/vi_project/bin/python program.py