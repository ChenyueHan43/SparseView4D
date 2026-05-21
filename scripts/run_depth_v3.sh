#!/bin/bash
#SBATCH --job-name=depth_v3
#SBATCH --account=torch_pr_926_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/ch5085/logs/depth_v3_%j.log

source ~/.bashrc
conda activate /scratch/ch5085/envs/deform3dgs
export TORCH_HOME=/scratch/ch5085/torch_cache
cd /scratch/ch5085/Deformable-3D-Gaussians

# 生成更稀疏的点云 (subsample=40, 约25000点)
for SCENE in jumpingjacks standup; do
    SCENE_DIR=/scratch/ch5085/data/data/$SCENE
    for N in 4 6 8; do
        python /scratch/ch5085/depth_init_v2.py \
            $SCENE_DIR \
            transforms_train_${N}views.json \
            ${SCENE_DIR}/points3d_depth_v3_${N}views.ply \
            40
    done
done

# 重跑 jumpingjacks
SCENE_DIR=/scratch/ch5085/data/data/jumpingjacks
for N in 4 6 8; do
    echo "=== jumpingjacks ${N}-view v3 ==="
    python train.py \
        -s $SCENE_DIR \
        -m /scratch/ch5085/output/jumpingjacks_${N}views_v3 \
        --eval --is_blender \
        --train_json transforms_train_${N}views.json \
        --init_ply ${SCENE_DIR}/points3d_depth_v3_${N}views.ply \
        --use_temporal_smooth \
        --lambda_temporal 0.01
    python render.py -m /scratch/ch5085/output/jumpingjacks_${N}views_v3 --mode render
    python metrics.py -m /scratch/ch5085/output/jumpingjacks_${N}views_v3
done

# 重跑 standup
SCENE_DIR=/scratch/ch5085/data/data/standup
for N in 4 6 8; do
    echo "=== standup ${N}-view v3 ==="
    python train.py \
        -s $SCENE_DIR \
        -m /scratch/ch5085/output/standup_${N}views_v3 \
        --eval --is_blender \
        --train_json transforms_train_${N}views.json \
        --init_ply ${SCENE_DIR}/points3d_depth_v3_${N}views.ply \
        --use_temporal_smooth \
        --lambda_temporal 0.01
    python render.py -m /scratch/ch5085/output/standup_${N}views_v3 --mode render
    python metrics.py -m /scratch/ch5085/output/standup_${N}views_v3
done

echo "=== All done ==="
