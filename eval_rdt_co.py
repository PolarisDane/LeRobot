from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus

leader_port = "/dev/ttyUSB0"
follower_port = "/dev/ttyUSB2"

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

robot = ManipulatorRobot(
    robot_type="koch",
    leader_arms={"left": leader_arm},
    follower_arms={"left": follower_arm},
    calibration_dir=".cache/calibration/koch",
    cameras={
        "view1": OpenCVCamera(12, fps=30, width=640, height=480), # rgb
        # "view2": OpenCVCamera(6, fps=30, width=640, height=480),
    },
)
robot.connect()
print("Robot connected")

import argparse
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_publish_step', action='store', type=int, 
                        help='Maximum number of action publishing steps', default=10000, required=False)
    parser.add_argument('--seed', action='store', type=int, 
                        help='Random seed', default=None, required=False)

    parser.add_argument('--img_front_topic', action='store', type=str, help='img_front_topic',
                        default='/camera_f/color/image_raw', required=False)
    parser.add_argument('--img_left_topic', action='store', type=str, help='img_left_topic',
                        default='/camera_l/color/image_raw', required=False)
    parser.add_argument('--img_right_topic', action='store', type=str, help='img_right_topic',
                        default='/camera_r/color/image_raw', required=False)
    
    parser.add_argument('--img_front_depth_topic', action='store', type=str, help='img_front_depth_topic',
                        default='/camera_f/depth/image_raw', required=False)
    parser.add_argument('--img_left_depth_topic', action='store', type=str, help='img_left_depth_topic',
                        default='/camera_l/depth/image_raw', required=False)
    parser.add_argument('--img_right_depth_topic', action='store', type=str, help='img_right_depth_topic',
                        default='/camera_r/depth/image_raw', required=False)
    
    parser.add_argument('--puppet_arm_left_cmd_topic', action='store', type=str, help='puppet_arm_left_cmd_topic',
                        default='/master/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_cmd_topic', action='store', type=str, help='puppet_arm_right_cmd_topic',
                        default='/master/joint_right', required=False)
    parser.add_argument('--puppet_arm_left_topic', action='store', type=str, help='puppet_arm_left_topic',
                        default='/puppet/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_topic', action='store', type=str, help='puppet_arm_right_topic',
                        default='/puppet/joint_right', required=False)
    
    parser.add_argument('--robot_base_topic', action='store', type=str, help='robot_base_topic',
                        default='/odom_raw', required=False)
    parser.add_argument('--robot_base_cmd_topic', action='store', type=str, help='robot_base_topic',
                        default='/cmd_vel', required=False)
    parser.add_argument('--use_robot_base', action='store_true', 
                        help='Whether to use the robot base to move around',
                        default=False, required=False)
    parser.add_argument('--publish_rate', action='store', type=int, 
                        help='The rate at which to publish the actions',
                        default=30, required=False)
    parser.add_argument('--ctrl_freq', action='store', type=int, 
                        help='The control frequency of the robot',
                        default=25, required=False)
    
    parser.add_argument('--chunk_size', action='store', type=int, 
                        help='Action chunk size',
                        default=64, required=False)
    parser.add_argument('--arm_steps_length', action='store', type=float, 
                        help='The maximum change allowed for each joint per timestep',
                        default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2], required=False)

    parser.add_argument('--use_actions_interpolation', action='store_true',
                        help='Whether to interpolate the actions if the difference is too large',
                        default=False, required=False)
    parser.add_argument('--use_depth_image', action='store_true', 
                        help='Whether to use depth images',
                        default=False, required=False)
    
    parser.add_argument('--disable_puppet_arm', action='store_true',
                        help='Whether to disable the puppet arm. This is useful for safely debugging',default=False)
    
    parser.add_argument('--config_path', type=str, default="configs/base.yaml", 
                        help='Path to the config file')
    # parser.add_argument('--cfg_scale', type=float, default=2.0,
    #                     help='the scaling factor used to modify the magnitude of the control features during denoising')
    parser.add_argument('--pretrained_model_name_or_path', type=str, required=True, help='Name or path to the pretrained model')
    
    parser.add_argument('--lang_embeddings_path', type=str, required=True, 
                        help='Path to the pre-encoded language instruction embeddings')
    
    args = parser.parse_args()
    return args


# from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.rdt.scripts.lca_inference import RDTInference
import time
import torch
from lerobot.scripts.control_robot import busy_wait
import numpy as np

inference_time_s = 60
fps = 30
device = "cuda" 

args = get_arguments()
policy = RDTInference(args)

# total_step = inference_time_s * fps
current_step = 0

pre_action= np.array([0, 120, 180, 0, 0, 0]) # TODO: initialize with the initial action
action = None

with torch.inference_mode():
    while  current_step < policy.max_publish_step:
        # Read the follower state and access the frames from the cameras
        observation = robot.capture_observation()

        # Convert to pytorch format: channel first and float32 in [0,1]
        # with batch dimension
        for name in observation:
            if "image" in name:
                # torch rgb images
                observation[name] = observation[name].type(torch.float32) / 255
                observation[name] = observation[name].permute(2, 0, 1).contiguous()
            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(device)

        policy.update_observation_window(observation)

        if current_step % policy.chunk_size == 0:
            action_buffer = policy.inference_fn(current_step) # .copy() # (64, 6)

        raw_action = action_buffer[current_step % policy.chunk_size]
        action = raw_action

        if args.use_actions_interpolation:
            # print(f"Time {t}, pre {pre_action}, act {action}")
            interp_actions = policy.interpolate_action(pre_action, action)
        else:
            interp_actions = action[np.newaxis, :]

        for act in interp_actions:
            # input("Press Enter to continue...")
            start_time = time.perf_counter()
            robot.send_action(act)
            dt_s = time.perf_counter() - start_time
            busy_wait(1 / fps - dt_s)

        current_step += 1
        pre_action = action

robot.disconnect()

print("eval done")

# for _ in range(inference_time_s * fps):
#     start_time = time.perf_counter()

#     # Read the follower state and access the frames from the cameras
#     observation = robot.capture_observation()

#     # Convert to pytorch format: channel first and float32 in [0,1]
#     # with batch dimension
#     for name in observation:
#         if "image" in name: # torch rgb images
#             observation[name] = observation[name].type(torch.float32) / 255
#             observation[name] = observation[name].permute(2, 0, 1).contiguous() # C, H, W
#         observation[name] = observation[name].unsqueeze(0)
#         observation[name] = observation[name].to(device)

#     # Compute the next action with the policy
#     # based on the current observation
#     action = policy.select_action(observation)
#     # Remove batch dimension
#     action = action.squeeze(0)
#     # Move to cpu, if not already the case
#     action = action.to("cpu")
#     # Order the robot to move
#     robot.send_action(action)

#     dt_s = time.perf_counter() - start_time
#     busy_wait(1 / fps - dt_s)