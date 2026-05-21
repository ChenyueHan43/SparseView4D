#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def kl_divergence(rho, rho_hat):
    rho_hat = torch.mean(torch.sigmoid(rho_hat), 0)
    rho = torch.tensor([rho] * len(rho_hat)).cuda()
    return torch.mean(
        rho * torch.log(rho / (rho_hat + 1e-5)) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat + 1e-5)))


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def find_knn_chunked(xyz, k=8, chunk_size=2000):
    """分块计算 KNN，避免 N×N OOM"""
    N = xyz.shape[0]
    idx_list = []
    for i in range(0, N, chunk_size):
        chunk = xyz[i:i+chunk_size]
        diff = chunk.unsqueeze(1) - xyz.unsqueeze(0)  # [C, N, 3]
        dist = (diff ** 2).sum(-1)  # [C, N]
        _, idx = dist.topk(k + 1, dim=1, largest=False)
        idx_list.append(idx[:, 1:])  # 去掉自身
    return torch.cat(idx_list, dim=0)  # [N, k]


def arap_loss(xyz, d_xyz, k=8):
    """
    As-Rigid-As-Possible 刚性约束
    要求相邻 Gaussian 之间的相对位置在变形前后保持一致
    xyz:   [N, 3] 变形前位置
    d_xyz: [N, 3] 变形量
    """
    import torch

    N = xyz.shape[0]
    xyz_deformed = xyz + d_xyz

    # 找 K 近邻
    with torch.no_grad():
        knn_idx = find_knn_chunked(xyz.detach(), k=k)  # [N, k]

    # 变形前相对位置
    xyz_j = xyz[knn_idx]           # [N, k, 3]
    xyz_i = xyz.unsqueeze(1).expand_as(xyz_j)
    e_before = xyz_i - xyz_j       # [N, k, 3]

    # 变形后相对位置
    xyz_def_j = xyz_deformed[knn_idx]
    xyz_def_i = xyz_deformed.unsqueeze(1).expand_as(xyz_def_j)
    e_after = xyz_def_i - xyz_def_j  # [N, k, 3]

    # 用 SVD 估计局部最优旋转 R_i
    # S = e_before^T @ e_after  -> [N, 3, 3]
    S = torch.bmm(
        e_before.transpose(1, 2),   # [N, 3, k]
        e_after                      # [N, k, 3]
    )  # [N, 3, 3]

    try:
        U, _, Vh = torch.linalg.svd(S)
        R = torch.bmm(Vh.transpose(-2, -1), U.transpose(-2, -1))  # [N, 3, 3]
    except Exception:
        # SVD 失败时退化为 L2
        loss = ((e_after - e_before) ** 2).sum(-1).mean()
        return loss

    # 刚性误差：||e_after - R @ e_before||^2
    e_before_rot = torch.bmm(
        e_before,           # [N, k, 3]
        R.transpose(-2, -1) # [N, 3, 3]
    )  # [N, k, 3]

    loss = ((e_after - e_before_rot) ** 2).sum(-1).mean()
    return loss
