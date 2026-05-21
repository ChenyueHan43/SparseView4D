#!/bin/bash
#SBATCH --job-name=trex_all
#SBATCH --account=torch_pr_926_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/ch5085/logs/trex_%j.log

source ~/.bashrc
conda activate /scratch/ch5085/envs/deform3dgs
export TORCH_HOME=/scratch/ch5085/torch_cache
cd /scratch/ch5085/Deformable-3D-Gaussians

SCENE_DIR=/scratch/ch5085/data/data/trex

# 生成 sparse json
python /scratch/ch5085/subsample_views.py $SCENE_DIR

# 生成 depth 点云
for N in 4 6 8; do
    python /scratch/ch5085/depth_init.py \
        $SCENE_DIR \
        transforms_train_${N}views.json \
        ${SCENE_DIR}/points3d_depth_${N}views.ply
done

# Baseline dense
python train.py \
    -s $SCENE_DIR \
    -m /scratch/ch5085/output/trex_dense \
    --eval --is_blender
python render.py -m /scratch/ch5085/output/trex_dense --mode render
python metrics.py -m /scratch/ch5085/output/trex_dense

# Baseline sparse
for N in 4 6 8; do
    echo "=== Baseline: ${N}-view ==="
    python train.py \
        -s $SCENE_DIR \
        -m /scratch/ch5085/output/trex_${N}views \
        --eval --is_blender \
        --train_json transforms_train_${N}views.json
    python render.py -m /scratch/ch5085/output/trex_${N}views --mode render
    python metrics.py -m /scratch/ch5085/output/trex_${N}views
done

# +Depth+Temporal
for N in 4 6 8; do
    echo "=== Full pipeline: ${N}-view ==="
    python train.py \
        -s $SCENE_DIR \
        -m /scratch/ch5085/output/trex_${N}views_full \
        --eval --is_blender \
        --train_json transforms_train_${N}views.json \
        --init_ply ${SCENE_DIR}/points3d_depth_${N}views.ply \
        --use_temporal_smooth \
        --lambda_temporal 0.01
    python render.py -m /scratch/ch5085/output/trex_${N}views_full --mode render
    python metrics.py -m /scratch/ch5085/output/trex_${N}views_full
done

echo "=== All done ==="
