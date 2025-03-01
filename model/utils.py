from collections import OrderedDict
import torchvision
import torch
import torch.nn as nn
import os
import model.networks as networks
from .base_model import BaseModel
import numpy as np
import random
from math import pi
import cv2

def warp_points(points, homographies, device='cuda'):
    """
    Warp a list of points with the given homography.

    Arguments:
        points: list of N points, shape (N, 2(x, y))).
        homography: batched or not (shapes (B, 3, 3) and (...) respectively).

    Returns: a Tensor of shape (N, 2) or (B, N, 2(x, y)) (depending on whether the homography
            is batched) containing the new coordinates of the warped points.

    """
    homographies =  homographies.float()
    no_batches = len(homographies.shape) == 2
    homographies = homographies.unsqueeze(0) if no_batches else homographies
    batch_size = homographies.shape[0]
    points = torch.cat((points.float(), torch.ones((points.shape[0], 1)).to(device)), dim=1)
    points = points.to(device)
    #homographies = homographies.view(batch_size*3,3)
    homographies = homographies.reshape(batch_size * 3, 3)
    warped_points = homographies@points.transpose(0,1)

    warped_points = warped_points.view([batch_size, 3, -1])
    warped_points = warped_points.transpose(2, 1)
    warped_points = warped_points[:, :, :2] / warped_points[:, :, 2:]
    return warped_points[0,:,:] if no_batches else warped_points
    
def apply_homography_vectorized(p1, H):
    with torch.no_grad():
        b, c, h, w = p1.shape
        # 创建网格坐标
        H_inv = torch.inverse(H)
        grid = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)), dim=2).to(p1.device)
        grid_positive = grid.clone()
        grid = grid.view([-1, 2])
        grid = torch.stack((grid[:, 1], grid[:, 0]), dim=1)
        # 广播单应矩阵并应用变换
        grid_new = warp_points(grid, H_inv, p1.device)
        grid_new = torch.stack((grid_new[:, :, 1], grid_new[:, :, 0]), dim=2)
        grid_new = grid_new.view(-1, h, w, 2).expand(4, -1, -1, -1)
        grid_new = grid_new.to(torch.int).float()

        mask = torch.zeros((b, h, w), device=p1.device, dtype=torch.uint8)
        for i in range(b):
            x_coords = grid_new[i, :, :, 0].floor().long()
            y_coords = grid_new[i, :, :, 1].floor().long()
            valid_mask = (x_coords >= 0) & (x_coords < w) & (y_coords >= 0) & (y_coords < h)
            mask[i] = valid_mask.float().view(h, w).round().byte()

        p1_t = torch.zeros_like(p1).to(p1.device)
        grid_new_positive = grid_new.clone()
        grid_new_positive[..., 0] = torch.clamp(grid_new_positive[..., 0], min=0, max=479)
        grid_new_positive[..., 1] = torch.clamp(grid_new_positive[..., 1], min=0, max=479)

        # 将 grid_new 的坐标转换为 p1 中的索引
        x = grid_positive[..., 0].long().unsqueeze(0).expand(4, -1, -1)
        y = grid_positive[..., 1].long().unsqueeze(0).expand(4, -1, -1)

        # 使用索引将 p1 中的值复制到 p2 中对应的位置
        #p1_t[torch.arange(b)[:, None, None], :, y, x] = p1[torch.arange(b)[:, None, None], :, indices_y, indices_x]
        for i in range(b):
            for j in range(c):
                x_coords1 = grid_new_positive[..., 0].long()
                y_coords1 = grid_new_positive[..., 1].long()
                p1_t[i, j, x, y] = p1[i, j, x_coords1, y_coords1]
    return p1_t, mask
    
def sample_homography_np(
        shape1, shape2, shift=0, perspective=True, scaling=True, rotation=True, translation=True,
        n_scales=5, n_angles=25, scaling_amplitude=0.2, perspective_amplitude_x=0.1,
        perspective_amplitude_y=0.1, patch_ratio=0.85, max_angle=pi/2,
        allow_artifacts=True, translation_overflow=0.):
    """Sample a random valid homography.

    Computes the homography transformation between a random patch in the original image
    and a warped projection with the same image size.
    As in `tf.contrib.image.transform`, it maps the output point (warped patch) to a
    transformed input point (original patch).
    The original patch, which is initialized with a simple half-size centered crop, is
    iteratively projected, scaled, rotated and translated.

    Arguments:
        shape: A rank-2 `Tensor` specifying the height and width of the original image.
        perspective: A boolean that enables the perspective and affine transformations.
        scaling: A boolean that enables the random scaling of the patch.
        rotation: A boolean that enables the random rotation of the patch.
        translation: A boolean that enables the random translation of the patch.
        n_scales: The number of tentative scales that are sampled when scaling.
        n_angles: The number of tentatives angles that are sampled when rotating.
        scaling_amplitude: Controls the amount of scale.
        perspective_amplitude_x: Controls the perspective effect in x direction.
        perspective_amplitude_y: Controls the perspective effect in y direction.
        patch_ratio: Controls the size of the patches used to create the homography.
        max_angle: Maximum angle used in rotations.
        allow_artifacts: A boolean that enables artifacts when applying the homography.
        translation_overflow: Amount of border artifacts caused by translation.

    Returns:
        A `Tensor` of shape `[1, 8]` corresponding to the flattened homography transform.
    """

    # print("debugging")


    # Corners of the output image
    pts1 = np.stack([[0., 0.], [0., 1.], [1., 1.], [1., 0.]], axis=0)
    # Corners of the input patch
    margin = (1 - patch_ratio) / 2
    pts2 = margin + np.array([[0, 0], [0, patch_ratio],
                                 [patch_ratio, patch_ratio], [patch_ratio, 0]])

    from numpy.random import normal
    from numpy.random import uniform
    from scipy.stats import truncnorm

    # Random perspective and affine perturbations
    # lower, upper = 0, 2
    std_trunc = 2

    if perspective:
        if not allow_artifacts:
            perspective_amplitude_x = min(perspective_amplitude_x, margin)
            perspective_amplitude_y = min(perspective_amplitude_y, margin)
        # perspective_displacement = tf.truncated_normal([1], 0., perspective_amplitude_y/2)
        # perspective_displacement = normal(0., perspective_amplitude_y/2, 1)
        perspective_displacement = truncnorm(-1*std_trunc, std_trunc, loc=0, scale=perspective_amplitude_y/2).rvs(1)
        # h_displacement_left = normal(0., perspective_amplitude_x/2, 1)
        h_displacement_left = truncnorm(-1*std_trunc, std_trunc, loc=0, scale=perspective_amplitude_x/2).rvs(1)
        # h_displacement_right = normal(0., perspective_amplitude_x/2, 1)
        h_displacement_right = truncnorm(-1*std_trunc, std_trunc, loc=0, scale=perspective_amplitude_x/2).rvs(1)
        pts2 += np.array([[h_displacement_left, perspective_displacement],
                          [h_displacement_left, -perspective_displacement],
                          [h_displacement_right, perspective_displacement],
                          [h_displacement_right, -perspective_displacement]]).squeeze()

    # Random scaling
    # sample several scales, check collision with borders, randomly pick a valid one
    if scaling:
        scales = truncnorm(-1*std_trunc, std_trunc, loc=1, scale=scaling_amplitude/2).rvs(n_scales)
        scales = np.concatenate((np.array([1]), scales), axis=0)

        # scales = np.concatenate( (np.ones((n_scales,1)), scales[:,np.newaxis]), axis=1)
        center = np.mean(pts2, axis=0, keepdims=True)
        scaled = (pts2 - center)[np.newaxis, :, :] * scales[:, np.newaxis, np.newaxis] + center
        if allow_artifacts:
            valid = np.arange(n_scales)  # all scales are valid except scale=1
        else:
            # valid = np.where((scaled >= 0.) * (scaled < 1.))
            valid = (scaled >= 0.) * (scaled < 1.)
            valid = valid.prod(axis=1).prod(axis=1)
            valid = np.where(valid)[0]
        idx = valid[np.random.randint(valid.shape[0], size=1)].squeeze().astype(int)
        pts2 = scaled[idx,:,:]

    # Random translation
    if translation:
        t_min, t_max = np.min(pts2, axis=0), np.min(1 - pts2, axis=0)
        if allow_artifacts:
            t_min += translation_overflow
            t_max += translation_overflow
        pts2 += np.array([uniform(-t_min[0], t_max[0],1), uniform(-t_min[1], t_max[1], 1)]).T

    # Random rotation
    # sample several rotations, check collision with borders, randomly pick a valid one
    if rotation:
        angles = np.linspace(-max_angle, max_angle, num=n_angles)
        angles = np.concatenate((angles, np.array([0.])), axis=0)  # in case no rotation is valid
        center = np.mean(pts2, axis=0, keepdims=True)
        rot_mat = np.reshape(np.stack([np.cos(angles), -np.sin(angles), np.sin(angles),
                                       np.cos(angles)], axis=1), [-1, 2, 2])
        rotated = np.matmul( (pts2 - center)[np.newaxis,:,:], rot_mat) + center
        if allow_artifacts:
            valid = np.arange(n_angles)  # all scales are valid except scale=1
        else:
            valid = (rotated >= 0.) * (rotated < 1.)
            valid = valid.prod(axis=1).prod(axis=1)
            valid = np.where(valid)[0]
        idx = valid[np.random.randint(valid.shape[0], size=1)].squeeze().astype(int)
        pts2 = rotated[idx,:,:]
        # idx = valid[tf.random_uniform((), maxval=tf.shape(valid)[0], dtype=tf.int32)]
        # pts2 = rotated[idx]

    # Rescale to actual size
    shape1 = shape1[::-1]  # different convention [y, x]
    pts11 =pts1 * shape1[np.newaxis,:]
    pts21 =pts2 * shape1[np.newaxis,:]
    shape2 = shape2[::-1]  # different convention [y, x]
    pts12 =pts1 * shape2[np.newaxis, :]
    pts22 =pts2 * shape2[np.newaxis, :]

    homography1 = cv2.getPerspectiveTransform(np.float32(pts11 + shift), np.float32(pts21 + shift))
    homography2 = cv2.getPerspectiveTransform(np.float32(pts12 + shift), np.float32(pts22 + shift))
    return homography1, homography2