#!/bin/sh
#SBATCH --account=ie-idi       
#SBATCH --job-name Yolov9s_hyp_aug
#SBATCH --time=0-01:00:00         # format: D-HH:MM:SS

#SBATCH --partition=GPUQ          # Asking for a GPU
#SBATCH --gres=gpu:1             # Setting the number of GPUs to 1
#SBATCH --mem=200G                 # Asking for 16GB RAM
#SBATCH --nodes=1
#SBATCH --output=/cluster/work/evaajo/vi_mini_project/Yolov9s_hyp_aug.txt    
#SBATCH --error=/cluster/work/evaajo/vi_mini_project/Yolov9s_hyp_aug_output.err       

#SBATCH --mail-user=evaajo@stud.ntnu.no
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
source activate miniproject_env
python /cluster/home/evaajo/vi_mini_project/program.py