#!/bin/bash
#SBATCH --job-name=rerender_v3
#SBATCH --account=torch_pr_926_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ch5085/logs/rerender_v3_%j.log

source ~/.bashrc
conda activate /scratch/ch5085/envs/deform3dgs
cd /scratch/ch5085/Deformable-3D-Gaussians

python render.py -m /scratch/ch5085/output/jumpingjacks_4views_v3 --mode render
python render.py -m /scratch/ch5085/output/jumpingjacks_8views_v3 --mode render

echo "Done"
