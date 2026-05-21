import json
import numpy as np
import os
import sys

def subsample_views(scene_dir, n_views, seed=42):
    input_path = os.path.join(scene_dir, 'transforms_train.json')
    with open(input_path) as f:
        data = json.load(f)
    
    frames = data['frames']
    total = len(frames)
    
    # 均匀采样
    np.random.seed(seed)
    indices = np.linspace(0, total - 1, n_views, dtype=int)
    sampled = [frames[i] for i in indices]
    
    out = {'camera_angle_x': data['camera_angle_x'], 'frames': sampled}
    
    output_path = os.path.join(scene_dir, f'transforms_train_{n_views}views.json')
    with open(output_path, 'w') as f:
        json.dump(out, f, indent=2)
    
    print(f'Saved {n_views} views to {output_path}')
    print(f'Indices: {indices.tolist()}')

if __name__ == '__main__':
    scene_dir = sys.argv[1]
    for n in [4, 6, 8]:
        subsample_views(scene_dir, n)
