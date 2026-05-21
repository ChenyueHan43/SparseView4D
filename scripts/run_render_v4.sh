#!/bin/bash
#SBATCH --job-name=render_v4
#SBATCH --account=torch_pr_926_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/ch5085/logs/render_v4_%j.log

source ~/.bashrc
conda activate /scratch/ch5085/envs/deform3dgs
cd /scratch/ch5085/Deformable-3D-Gaussians

python render.py -m /scratch/ch5085/output/jumpingjacks_4views_v4 --mode render

echo "Done"
