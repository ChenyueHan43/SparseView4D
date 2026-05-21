#!/bin/bash
#SBATCH --job-name=standup_8v
#SBATCH --account=torch_pr_926_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/ch5085/logs/standup_8v_%j.log

source ~/.bashrc
conda activate /scratch/ch5085/envs/deform3dgs
export TORCH_HOME=/scratch/ch5085/torch_cache
cd /scratch/ch5085/Deformable-3D-Gaussians

SCENE_DIR=/scratch/ch5085/data/data/standup

python train.py \
    -s $SCENE_DIR \
    -m /scratch/ch5085/output/standup_8views_full \
    --eval --is_blender \
    --train_json transforms_train_8views.json \
    --init_ply ${SCENE_DIR}/points3d_depth_v2_8views.ply \
    --use_temporal_smooth \
    --lambda_temporal 0.01

python render.py -m /scratch/ch5085/output/standup_8views_full --mode render
python metrics.py -m /scratch/ch5085/output/standup_8views_full

echo "=== Done ==="
