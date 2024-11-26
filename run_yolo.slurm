#!/bin/sh
#SBATCH --account=ie-idi       # E.g. "ie-idi" if you belong to IDI
#SBATCH --job-name=VI-miniproject-adam-9-cut
#SBATCH --time=0-10:00:00         # format: D-HH:MM:SS

#SBATCH --partition=GPUQ          # Asking for a GPU
#SBATCH --gres=gpu:1             # Setting the number of GPUs to 1
#SBATCH --mem-per-gpu=16G                 # Asking for 16GB RAM
#SBATCH --nodes=1
#SBATCH --output=output/VI-miniproject-adam-9-cut.txt      # Specifying 'stdout'
#SBATCH --error=output/VI-miniproject-adam-9-cut.err        # Specifying 'stderr'

#SBATCH --mail-user=ingval@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

module purge

# Running your python file
module load Anaconda3/2024.02-1
source activate vi_project

python program.py