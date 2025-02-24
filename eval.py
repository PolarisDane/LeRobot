from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus
import numpy as np
import open3d as o3d

leader_port = "/dev/ttyUSB1"
follower_port = "/dev/ttyUSB0"

leader_arm = DynamixelMotorsBus(
    port=leader_port,
    motors={
        # name: (index, model)
        "shoulder_pan": (1, "xl330-m077"),
        "shoulder_lift": (2, "xl330-m077"),
        "elbow_flex": (3, "xl330-m077"),
        "wrist_flex": (4, "xl330-m077"),
        "wrist_roll": (5, "xl330-m077"),
        "gripper": (6, "xl330-m077"),
    },
)

follower_arm = DynamixelMotorsBus(
    port=follower_port,
    motors={
        # name: (index, model)
        "shoulder_pan": (1, "xl430-w250"),
        "shoulder_lift": (2, "xl430-w250"),
        "elbow_flex": (3, "xl330-m288"),
        "wrist_flex": (4, "xl330-m288"),
        "wrist_roll": (5, "xl330-m288"),
        "gripper": (6, "xl330-m288"),
    },
)

from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera
from lerobot.common.robot_devices.cameras.intelrealsense import IntelRealSenseCamera

robot = ManipulatorRobot(
    robot_type="koch",
    leader_arms={"left": leader_arm},
    follower_arms={"left": follower_arm},
    calibration_dir=".cache/calibration/koch",
    cameras={
        "view_front": IntelRealSenseCamera(408122072021,fps=30, width=640, height=480, use_depth = True),
        "view_side": IntelRealSenseCamera(408322070928,fps=30, width=640, height=480, use_depth = True),
    },
)
robot.connect()

# from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.atm.modeling_atm import BCViLTPolicy
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.SNN.modeling_snn import SNNPolicy
from lerobot.common.policies.spikeact.modeling_spikeact import SpikeACTPolicy
from lerobot.common.policies.vqbet.modeling_vqbet import VQBeTPolicy
from lerobot.common.policies.RISE.modeling_RISE import RISEPolicy
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
import time
import torch
import cv2
from lerobot.scripts.control_robot import busy_wait
from safetensors import safe_open

inference_time_s = 60
fps = 30
device = "cuda"  # TODO: On Mac, use "mps" or "cpu"

def unflatten_dict(d, sep="/"):
    outdict = {}
    for key, value in d.items():
        parts = key.split(sep)
        d = outdict
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return outdict

def create_infer_policy(stats_file, policy_cls):
    tensors = {}
    with safe_open(stats_file, framework="pt", device=0) as fn:
        for key in fn.keys():
            tensors[key] = fn.get_tensor(key)
    # import pdb; pdb.set_trace()
    kwargs = {}
    kwargs["dataset_stats"] = unflatten_dict(tensors)
    policy = policy_cls.from_pretrained("lerobot/pi0", **kwargs)
    return policy

ckpt_path = "/home/rhos/lerobot/tmp/RISE_2025_2_21_t10003_limited_randomization/checkpoints/last/pretrained_model"
policy = create_infer_policy("/home/rhos/lerobot/data/t20005_limited_randomization/meta_data/stats.safetensors", PI0Policy)
# policy = RISEPolicy.from_pretrained(ckpt_path)
policy.eval()
policy.to(device)

camera_intrinsic = np.array([[604.534,0,313.181],[0,604.168,250.928],[0,0,1]])

def load_point_cloud(rgb_image, depth_image, voxel_size):
        # 相机内参（根据你的相机进行设置）
        fx = camera_intrinsic[0, 0]
        fy = camera_intrinsic[1, 1]
        cx = camera_intrinsic[0, 2]
        cy = camera_intrinsic[1, 2]
        scale = 1000  # 深度图像的比例因子

        # print(depth_image.shape)
        # torch.squeeze(depth_image, 0)
        # print(depth_image.shape)
        # depth_image = depth_image[0]
        # print(depth_image.shape)
        height, width = depth_image.shape
        rgb_image=rgb_image*255
        rgb_image=np.asarray(rgb_image.numpy().transpose((1,2,0)).astype(np.uint8),order='C')
        colors = o3d.geometry.Image(rgb_image)
        depths= o3d.geometry.Image(depth_image.numpy().astype(np.float32))
        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(colors, depths, scale, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsics)

        pcd = pcd.voxel_down_sample(voxel_size)
        
        points = np.array(pcd.points)
        colors = np.array(pcd.colors)
        return points.astype(np.float32), colors.astype(np.float32)

use_pointcloud = isinstance(policy, RISEPolicy)
print("use_pointcloud:", use_pointcloud)
for _ in range(inference_time_s * fps):
    start_time = time.perf_counter()

    # Read the follower state and access the frames from the cameras
    observation = robot.capture_observation()

    # Convert to pytorch format: channel first and float32 in [0,1]
    # with batch dimension
    for name in observation:
        if "image" in name:
            observation[name] = observation[name].type(torch.float32) / 255
            observation[name] = observation[name].permute(2, 0, 1).contiguous()
        observation[name] = observation[name].unsqueeze(0)
        observation[name] = observation[name].to(device)

    if use_pointcloud:
            clouds=[]
            expected_image_keys = ["observation.images.view_front"]
            expected_depth_keys = ["observation.depths.view_front"]

            images = torch.stack([observation[k] for k in expected_image_keys], dim=-4) # 1,v,c,h,w
            depths = torch.stack([observation[k] for k in expected_depth_keys], dim=-3) # 1,v,h,w
            print(images.shape)
            voxel_size=0.005

            for i in range(images.shape[0]):
                points, colors = load_point_cloud(images[i,0].cpu(), depths[i,0].cpu(),voxel_size)
            #TODO: apply imagenet normalization
            cloud = np.concatenate([points, colors], axis = -1)
            clouds.append(cloud)

            input_coords_list = []
            input_feats_list = []
            for cloud in clouds:
                # Upd Note. Make coords contiguous.
                coords = np.ascontiguousarray(cloud[:, :3] / voxel_size, dtype = np.int32)
                # Upd Note. API change.
                input_coords_list.append(coords)
                input_feats_list.append(cloud.astype(np.float32))

            observation['input_coords_list'] = [input_coords_list]
            observation['input_feats_list'] = [input_feats_list]
    # Compute the next action with the policy
    # based on the current observation
    observation['task'] = ["Push the cube exactly into the white area."]
    # import pdb; pdb.set_trace()
    action = policy.select_action(observation)
    # Remove batch dimension
    action = action.squeeze(0)
    # Move to cpu, if not already the case
    action = action.to("cpu")
    # follower_pos_1 = robot.follower_arms["left"].read("Present_Position")

    # print("action: ", action)

    # print("real: ", follower_pos_1)

    # Order the robot to move
    print(action)
    real_action = robot.send_action(action)

    # print(torch.equal(action, real_action))

    dt_s = time.perf_counter() - start_time
    busy_wait(1 / fps - dt_s)

robot.disconnect()

print("eval done")
