#!/bin/bash
#SBATCH --job-name=keyframe_transition
#SBATCH --time=05:00:00
#SBATCH --partition=plgrid-gpu-v100
#SBATCH --mem=40G
#SBATCH --gres=gpu:1  # Explicitly request 1 GPU
#SBATCH -A plglscclass24-gpu
#SBATCH --cpus-per-task=4

module load python
module load pytorch/1.12.1-foss-2021a-cuda-11.3.1
# Enable CUDA debugging
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0
echo $CUDA_VISIBLE_DEVICES
# Debug GPU allocation
srun nvidia-smi

# Run the script
srun python train_context_model.py lafan1_context_model_ending_transition
