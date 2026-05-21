#!/bin/bash
#SBATCH --job-name=multiframe
#SBATCH --account=torch_pr_926_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/ch5085/logs/multiframe_%j.log

source ~/.bashrc
conda activate /scratch/ch5085/envs/deform3dgs
export TORCH_HOME=/scratch/ch5085/torch_cache
cd /scratch/ch5085/Deformable-3D-Gaussians

# 重新编译 submodules for l40s
rm -rf submodules/simple-knn/build submodules/depth-diff-gaussian-rasterization/build
TORCH_CUDA_ARCH_LIST="8.9" CUDA_HOME=/usr/local/cuda MAX_JOBS=4 \
    pip install submodules/simple-knn submodules/depth-diff-gaussian-rasterization \
    --no-build-isolation --force-reinstall --no-cache-dir \
    --cache-dir /scratch/ch5085/pip_cache -q

SCENE_DIR=/scratch/ch5085/data/data/jumpingjacks

for N in 4 6 8; do
    echo "=== Multiframe: ${N}-view ==="
    python train.py \
        -s $SCENE_DIR \
        -m /scratch/ch5085/output/jumpingjacks_${N}views_mf \
        --eval --is_blender \
        --train_json transforms_train_${N}views.json \
        --init_ply ${SCENE_DIR}/points3d_depth_${N}views.ply \
        --use_multiframe \
        --lambda_multiframe 0.5

    python render.py \
        -m /scratch/ch5085/output/jumpingjacks_${N}views_mf \
        --mode render

    python metrics.py \
        -m /scratch/ch5085/output/jumpingjacks_${N}views_mf
done

echo "=== All done ==="
