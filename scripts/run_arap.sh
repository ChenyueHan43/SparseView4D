#!/bin/bash
#SBATCH --job-name=arap
#SBATCH --account=torch_pr_926_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/ch5085/logs/arap_%j.log

source ~/.bashrc
conda activate /scratch/ch5085/envs/deform3dgs
export TORCH_HOME=/scratch/ch5085/torch_cache
cd /scratch/ch5085/Deformable-3D-Gaussians

SCENE_DIR=/scratch/ch5085/data/data/jumpingjacks

for N in 4 6 8; do
    echo "=== ARAP: ${N}-view ==="
    python train.py \
        -s $SCENE_DIR \
        -m /scratch/ch5085/output/jumpingjacks_${N}views_arap \
        --eval --is_blender \
        --train_json transforms_train_${N}views.json \
        --init_ply ${SCENE_DIR}/points3d_depth_${N}views.ply \
        --use_temporal_smooth \
        --lambda_temporal 0.01 \
        --use_arap \
        --lambda_arap 0.01

    python render.py \
        -m /scratch/ch5085/output/jumpingjacks_${N}views_arap \
        --mode render

    python metrics.py \
        -m /scratch/ch5085/output/jumpingjacks_${N}views_arap
done

echo "=== All done ==="
