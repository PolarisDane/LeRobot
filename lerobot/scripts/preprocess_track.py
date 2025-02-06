import json
import os
from glob import glob

import click
import h5py
import numpy as np
import torch
from einops import rearrange
from natsort import natsorted
from tqdm import tqdm
from PIL import Image
from transformers import AutoTokenizer, AutoModel
from hydra.utils import to_absolute_path
import imageio.v3 as iio

from lerobot.common.policies.atm.utils.flow_utils import sample_from_mask, sample_double_grid

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# EXTRA_STATES_KEYS = ['gripper_states', 'joint_states', 'ee_ori', 'ee_pos', 'ee_states']




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

    # sample random points
    points = sample_from_mask(np.ones((H, W, 1)) * 255, num_samples=num_points)
    points = torch.from_numpy(points).float().cuda()
    points = torch.cat([torch.randint_like(points[:, :1], 0, T), points], dim=-1).cuda()

    # sample grid points
    grid_points = sample_double_grid(7, device="cuda")
    grid_points[:, 0] = grid_points[:, 0] * H
    grid_points[:, 1] = grid_points[:, 1] * W
    grid_points = torch.cat([torch.randint_like(grid_points[:, :1], 0, T), grid_points], dim=-1).cuda()

    pred_tracks, pred_vis = track_and_remove(track_model, video[None], points[None],var_threshold=1.)
    pred_grid_tracks, pred_grid_vis = track_and_remove(track_model, video[None], grid_points[None], var_threshold=0.)

    pred_tracks = torch.cat([pred_grid_tracks, pred_tracks], dim=2)
    pred_vis = torch.cat([pred_grid_vis, pred_vis], dim=2)
    return pred_tracks, pred_vis


def collect_states_from_demo(h5_file, video, track_model, num_points):
    # print(video.shape)

    root_group = h5_file.create_group("root") if "root" not in h5_file else h5_file["root"]


    view = "view_0"

    T, C, H, W = video.shape

    pred_tracks, pred_vis = track_through_video(video, track_model, num_points=num_points)

    # [1, T, N, 2], normalize coordinates to [0, 1] for in-picture coordinates
    pred_tracks[:, :, :, 0] /= W
    pred_tracks[:, :, :, 1] /= H

    view_group = root_group.create_group(view) if view not in root_group else root_group[view]

    # we only have one view for real human video data, so we manually set the view group and name it view_0

    # if real robot data with wrist camera is used, then we need a loop here to deal with each view

    if "video" not in view_group:
        view_group.create_dataset("video", data=video.cpu().numpy()[None].astype(np.uint8))

    if "tracks" in view_group:
        view_group.__delitem__("tracks")
    if "vis" in view_group:
        view_group.__delitem__("vis")
    view_group.create_dataset("tracks", data=pred_tracks.cpu().numpy())
    view_group.create_dataset("vis", data=pred_vis.cpu().numpy())


def save_images(video, image_dir, view):
    os.makedirs(image_dir, exist_ok=True)
    for idx, img in enumerate(video):
        img = img.astype(np.uint8)
        Image.fromarray(img).save(os.path.join(image_dir, f"{view}_{idx}.png"))

def initial_save_h5(path, skip_exist):
    if os.path.exists(path):
        with h5py.File(path, 'r') as f:
            if skip_exist:
                return None
            
    f = h5py.File(path, 'w')
    return f

def generate_data(source_video_path, target_dir, track_model, skip_exist):
    video_files = glob(os.path.join(source_video_path, "*.mp4"))
    video_files = natsorted(video_files)

    # setup visualization class

    num_points = 1000
    with torch.no_grad():
        for idx in tqdm(range(len(video_files))):
            video_file = video_files[idx]
            print(f"Processing {video_file}...")
            frames = iio.imread(video_file, plugin="FFMPEG")
            video = torch.tensor(frames).permute(0, 3, 1, 2).float().cuda()
            save_path = os.path.join(target_dir, f"{os.path.basename(video_file)}.h5")
            h5_file_handle = initial_save_h5(save_path, skip_exist)
            image_save_dir = os.path.join(target_dir, "images", os.path.basename(video_file).split('.')[0])
            print("image_save_dir:", image_save_dir)

            if h5_file_handle is None:
                continue

            # try:
            collect_states_from_demo(h5_file_handle, video, track_model, num_points)
            print(f"{video_file} is completed.")
            # except Exception as e:
            #     print(f"Exception {e} when processing {video_file}")
            #     exit()

@click.command()
@click.option("--video_dir", type=str, default="./data/videos/")
@click.option("--save_dir", type=str, default="./data/datasets/")
@click.option("--skip_exist", type=bool, default=False)
def main(video_dir, save_dir, skip_exist):
    """
    video_dir: str, the directory of original videos
    save_dir: str, the directory to save the preprocessed data
    skip_exist: bool, whether to skip the existing preprocessed videos
    """

    # setup cotracker
    cotracker = torch.hub.load(os.path.join(os.path.expanduser("~"), ".cache/torch/hub/facebookresearch_co-tracker_main/"), "cotracker3_offline", source="local")
    cotracker = cotracker.eval().cuda()

    # load task name embeddings

    source_video_path = video_dir
 
    task_dir = os.path.join(save_dir)
    os.makedirs(task_dir, exist_ok=True)
    generate_data(source_video_path, task_dir, cotracker,  skip_exist)


if __name__ == "__main__":
    main()
