from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus

leader_port_1 = "/dev/ttyUSB0"
follower_port_1 = "/dev/ttyUSB1"
#leader_port_2 = "/dev/ttyUSB0"
#follower_port_2 = "/dev/ttyUSB1"

leader_arm_1 = DynamixelMotorsBus(
    port=leader_port_1,
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

follower_arm_1 = DynamixelMotorsBus(
    port=follower_port_1,
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
'''
leader_arm_2 = DynamixelMotorsBus(
    port=leader_port_2,
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

follower_arm_2 = DynamixelMotorsBus(
    port=follower_port_2,
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
'''
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera

# robot = ManipulatorRobot(
#     robot_type="koch",
#     leader_arms={"left": leader_arm_1, "right": leader_arm_2},
#     follower_arms={"left": follower_arm_1, "right": follower_arm_2},
#     calibration_dir=".cache/calibration/koch_bimanual",
#     cameras={
#         "view1": OpenCVCamera(14, fps=30, width=640, height=480),
#         # "view2": OpenCVCamera(6, fps=30, width=640, height=480),
#     },
# )
robot = ManipulatorRobot(
    robot_type="koch",
    leader_arms={"left": leader_arm_1},
    follower_arms={"left": follower_arm_1},
    calibration_dir=".cache/calibration/koch",
    # cameras={
    #     "view1": OpenCVCamera(0, fps=30, width=640, height=480),
    #     # "view2": OpenCVCamera(6, fps=30, width=640, height=480),
    # },
)
robot.connect()

import time
from lerobot.scripts.control_robot import busy_wait

record_time_s = 100
fps = 60

# states = []
# actions = []
for _ in range(record_time_s * fps):
    start_time = time.perf_counter()
    observation, action = robot.teleop_step(record_data=True)

    # states.append(observation["observation.state"])
    # actions.append(action["action"])

    leader_pos_1 = robot.leader_arms["left"].read("Present_Position")
    follower_pos_1 = robot.follower_arms["left"].read("Present_Position")

    # leader_pos_2 = robot.leader_arms["right"].read("Present_Position")
    # follower_pos_2 = robot.follower_arms["right"].read("Present_Position")

    # print(leader_pos_1)
    print(follower_pos_1)

    # print(leader_pos_2)
    # print(follower_pos_2)

    dt_s = time.perf_counter() - start_time
    busy_wait(1 / fps - dt_s)

# print(states)

# print(actions)

robot.disconnect()

'''
python lerobot/scripts/control_robot.py record \
  --robot-path lerobot/configs/robot/koch.yaml \
  --fps 30 \
  --root . \
  --repo-id data/koch_test_2024_12_11 \
  --warmup-time-s 5 \
  --episode-time-s 30 \
  --reset-time-s 30 \
  --num-episodes 1 \
  --push-to-hub 0
'''

'''
DATA_DIR=data python lerobot/scripts/train.py \
  dataset_repo_id=koch_test_2024_11_06 \
  policy=dp_koch_real \
  env=koch_real \
  hydra.run.dir=outputs/train/dp_koch_test_2024_11_27_2 \
  hydra.job.name=dp_koch_test_2024_11_27_2 \
  device=cuda \
  wandb.enable=true \
  resume=true
'''



