import json
import os
import sys
import numpy as np
from PIL import Image
import torch
from transformers import pipeline
from plyfile import PlyData, PlyElement

def load_cameras(scene_dir, train_json):
    with open(os.path.join(scene_dir, train_json)) as f:
        data = json.load(f)
    fov_x = data['camera_angle_x']
    frames = data['frames']
    return fov_x, frames

def depth_to_pointcloud(depth, fov_x, c2w, max_depth=10.0):
    H, W = depth.shape
    fx = W / (2 * np.tan(fov_x / 2))
    fy = fx
    cx, cy = W / 2, H / 2

    # 反投影
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # 过滤无效点
    mask = (z > 0) & (z < max_depth)
    pts_cam = np.stack([x[mask], y[mask], z[mask]], axis=-1)

    # 相机坐标系转世界坐标系
    pts_hom = np.concatenate([pts_cam, np.ones((len(pts_cam), 1))], axis=-1)
    pts_world = (c2w @ pts_hom.T).T[:, :3]
    return pts_world

def scale_align_depth(depth, c2w, fov_x, sparse_pts, n_sample=500):
    """用稀疏随机点做 scale+shift 对齐"""
    if sparse_pts is None or len(sparse_pts) == 0:
        return depth

    H, W = depth.shape
    fx = W / (2 * np.tan(fov_x / 2))
    cx, cy = W / 2, H / 2

    # 把世界坐标的稀疏点投影到相机坐标
    w2c = np.linalg.inv(c2w)
    pts_hom = np.concatenate([sparse_pts, np.ones((len(sparse_pts), 1))], axis=-1)
    pts_cam = (w2c @ pts_hom.T).T
    valid = pts_cam[:, 2] > 0
    pts_cam = pts_cam[valid]

    if len(pts_cam) < 5:
        return depth

    # 投影到像素坐标
    u = (pts_cam[:, 0] * fx / pts_cam[:, 2] + cx).astype(int)
    v = (pts_cam[:, 1] * fx / pts_cam[:, 2] + cy).astype(int)
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v = u[in_bounds], v[in_bounds]
    z_gt = pts_cam[in_bounds, 2]
    z_pred = depth[v, u]

    valid2 = z_pred > 0
    if valid2.sum() < 3:
        return depth

    # 最小二乘 scale+shift
    A = np.stack([z_pred[valid2], np.ones(valid2.sum())], axis=-1)
    b = z_gt[valid2]
    result = np.linalg.lstsq(A, b, rcond=None)
    scale, shift = result[0]
    scale = max(scale, 0.01)  # 防止负数
    return depth * scale + shift

def generate_depth_pointcloud(scene_dir, train_json, output_ply, subsample=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading Depth Anything V2...")
    pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=0 if device == "cuda" else -1
    )

    fov_x, frames = load_cameras(scene_dir, train_json)
    all_points = []

    # 先生成随机稀疏点用于 scale align
    np.random.seed(42)
    sparse_ref = np.random.random((1000, 3)) * 2.6 - 1.3

    print(f"Processing {len(frames)} frames...")
    for i, frame in enumerate(frames):
        img_path = os.path.join(scene_dir, frame['file_path'] + '.png')
        if not os.path.exists(img_path):
            img_path = os.path.join(scene_dir, frame['file_path'])
        if not os.path.exists(img_path):
            print(f"  Skipping missing image: {img_path}")
            continue

        image = Image.open(img_path).convert('RGB')
        c2w = np.array(frame['transform_matrix'])

        # 估计深度
        result = pipe(image)
        depth = np.array(result['depth'])
        depth = depth.astype(np.float32)
        # Depth Anything 输出是相对深度，归一化到 0-1，做 scale align
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth = scale_align_depth(depth, c2w, fov_x, sparse_ref)

        # 反投影，每帧只取部分像素
        pts = depth_to_pointcloud(depth, fov_x, c2w)
        if subsample > 1 and len(pts) > 0:
            idx = np.random.choice(len(pts), min(len(pts), len(pts)//subsample), replace=False)
            pts = pts[idx]
        all_points.append(pts)
        print(f"  Frame {i+1}/{len(frames)}: {len(pts)} points")

    all_points = np.concatenate(all_points, axis=0)
    print(f"Total points: {len(all_points)}")

    # 写 ply
    colors = np.ones_like(all_points) * 128
    vertex = np.array(
        [tuple(p) + (0.0, 0.0, 0.0) + tuple(c) for p, c in zip(all_points, colors)],
        dtype=[('x','f4'),('y','f4'),('z','f4'),('nx','f4'),('ny','f4'),('nz','f4'),('red','u1'),('green','u1'),('blue','u1')]
    )
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el]).write(output_ply)
    print(f"Saved to {output_ply}")

if __name__ == '__main__':
    scene_dir = sys.argv[1]
    train_json = sys.argv[2]  # e.g. transforms_train_4views.json
    output_ply = sys.argv[3]  # e.g. points3d_depth_4views.ply
    generate_depth_pointcloud(scene_dir, train_json, output_ply)
