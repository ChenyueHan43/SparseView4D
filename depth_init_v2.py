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

def clip_depth_percentile(depth, lo=5, hi=95):
    """去掉最近和最远的异常点"""
    valid = depth > 0
    if valid.sum() == 0:
        return depth
    lo_val = np.percentile(depth[valid], lo)
    hi_val = np.percentile(depth[valid], hi)
    depth = np.clip(depth, lo_val, hi_val)
    return depth

def depth_to_pointcloud(depth, fov_x, c2w, max_depth=10.0):
    H, W = depth.shape
    fx = W / (2 * np.tan(fov_x / 2))
    cx, cy = W / 2, H / 2

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fx

    mask = (z > 0) & (z < max_depth)
    pts_cam = np.stack([x[mask], y[mask], z[mask]], axis=-1)

    pts_hom = np.concatenate([pts_cam, np.ones((len(pts_cam), 1))], axis=-1)
    pts_world = (c2w @ pts_hom.T).T[:, :3]
    return pts_world

def scale_align_from_cameras(depth, c2w, fov_x, all_c2w_list):
    """
    用相机间距推算场景尺度做 scale align
    思路：相机到场景中心的距离作为深度参考
    """
    # 所有相机位置
    cam_positions = np.array([m[:3, 3] for m in all_c2w_list])
    scene_center = cam_positions.mean(axis=0)
    cam_pos = c2w[:3, 3]
    dist_to_center = np.linalg.norm(cam_pos - scene_center)

    if dist_to_center < 1e-6:
        return depth

    # 当前深度图的中位数对齐到相机到场景中心的距离
    valid = depth > 0
    if valid.sum() == 0:
        return depth
    depth_median = np.median(depth[valid])
    if depth_median < 1e-6:
        return depth

    scale = dist_to_center / depth_median
    return depth * scale

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
    all_c2w = [np.array(f['transform_matrix']) for f in frames]
    all_points = []

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
        depth = np.array(result['depth']).astype(np.float32)

        # 归一化
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        # percentile clipping
        depth = clip_depth_percentile(depth, lo=5, hi=95)

        # scale align（用相机间距）
        depth = scale_align_from_cameras(depth, c2w, fov_x, all_c2w)

        # 反投影
        H, W = depth.shape
        fx = W / (2 * np.tan(fov_x / 2))
        cx, cy = W / 2, H / 2
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        z = depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fx
        mask = (z > 0) & (z < 20.0)
        pts_cam = np.stack([x[mask], y[mask], z[mask]], axis=-1)
        pts_hom = np.concatenate([pts_cam, np.ones((len(pts_cam), 1))], axis=-1)
        pts_world = (c2w @ pts_hom.T).T[:, :3]

        if subsample > 1 and len(pts_world) > 0:
            idx = np.random.choice(len(pts_world), len(pts_world)//subsample, replace=False)
            pts_world = pts_world[idx]

        all_points.append(pts_world)
        print(f"  Frame {i+1}/{len(frames)}: {len(pts_world)} points")

    all_points = np.concatenate(all_points, axis=0)
    print(f"Total points: {len(all_points)}")

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
    train_json = sys.argv[2]
    output_ply = sys.argv[3]
    subsample = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    generate_depth_pointcloud(scene_dir, train_json, output_ply, subsample=subsample)
