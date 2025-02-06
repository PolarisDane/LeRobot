import json
import os
from glob import glob

import click
import h5py
import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm
from PIL import Image
from transformers import AutoTokenizer, AutoModel
from hydra.utils import to_absolute_path

from lerobot.common.policies.atm.utils.flow_utils import sample_from_mask, sample_double_grid


def track_and_remove(tracker, video, points, var_threshold=10.):
    B, T, C, H, W = video.shape
    pred_tracks, pred_vis = tracker(video, queries=points, backward_tracking=True) # [1, T, N, 2]

    var = torch.var(pred_tracks, dim=1)  # [1, N, 2]
    var = torch.sum(var, dim=-1)[0]  # List

    # get index of points with low variance
    idx = torch.where(var > var_threshold)[0]
    if len(idx) == 0:
        print(torch.max(var))
        assert len(idx) > 0, 'No points with low variance'

    new_points = points[:, idx].clone()

    # Repeat and sample
    rep = points.shape[1] // len(idx) + 1
    new_points = torch.tile(new_points, (1, rep, 1))
    new_points = new_points[:, :points.shape[1]]
    # Add 10 percent height and width as noise
    noise = torch.randn_like(new_points[:, :, 1:]) * 0.05 * H
    new_points[:, :, 1:] += noise

    # Track new points
    pred_tracks, pred_vis = tracker(video, queries=new_points, backward_tracking=True)

    return pred_tracks, pred_vis


def track_through_video(video, track_model, num_points=1000):
    T, C, H, W = video.shape

    video = torch.from_numpy(video).cuda().float()

    # sample random points
    points = sample_from_mask(np.ones((H, W, 1)) * 255, num_samples=num_points)
    points = torch.from_numpy(points).float().cuda()
    points = torch.cat([torch.randint_like(points[:, :1], 0, T), points], dim=-1).cuda()

    # sample grid points
    grid_points = sample_double_grid(7, device="cuda")
    grid_points[:, 0] = grid_points[:, 0] * H
    grid_points[:, 1] = grid_points[:, 1] * W
    grid_points = torch.cat([torch.randint_like(grid_points[:, :1], 0, T), grid_points], dim=-1).cuda()

    pred_tracks, pred_vis = track_and_remove(track_model, video[None], points[None])
    pred_grid_tracks, pred_grid_vis = track_and_remove(track_model, video[None], grid_points[None], var_threshold=0.)

    pred_tracks = torch.cat([pred_grid_tracks, pred_tracks], dim=2)
    pred_vis = torch.cat([pred_grid_vis, pred_vis], dim=2)
    return pred_tracks, pred_vis


def collect_states_from_demo(rgb, demo_k, view_names, track_model, num_points):
    '''
    input:
    rgb: (T, H, W, C)

    returns:
    pred_tracks: (1, T, N, 2)
    pred_vis: (1, T, N)
    '''

    rgb = rearrange(rgb, "t h w c -> t c h w")
    T, C, H, W = rgb.shape

    pred_tracks, pred_vis = track_through_video(rgb, track_model, num_points=num_points)

    # [1, T, N, 2], normalize coordinates to [0, 1] for in-picture coordinates
    pred_tracks[:, :, :, 0] /= W
    pred_tracks[:, :, :, 1] /= H

    return pred_tracks, pred_vis
