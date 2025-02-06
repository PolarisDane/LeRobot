from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus
import numpy as np
from scipy.linalg import lstsq
import pybullet as p
import pybullet_data
import numpy as np

follower_port = "/dev/ttyUSB1"

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
    follower_arms={"main": follower_arm},
    calibration_dir=".cache/calibration/koch",
    cameras={
        "view1": OpenCVCamera(12, fps=30, width=640, height=480),
    },
)
robot.connect()
from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
robot.follower_arms["main"].write("Torque_Enable", TorqueMode.ENABLED.value)
import time
from lerobot.scripts.control_robot import busy_wait

record_time_s = 30
fps = 60

states = []
actions = []


base_points_camera = np.array([
    [0.10544403, -0.18065751, 0.55699998],  # 第一个基座点
    [0.09391508, -0.16818333, 0.55000001],  # 第二个基座点
    [0.11092141, -0.16563238, 0.55000001]   # 第三个基座点
])

target_position_camera = np.array([0.00643117, 0.01511228, 0.375])


def calculate_local_coordinate_system(base_points_camera):
    p1, p2, p3 = base_points_camera

    z_axis = np.cross(p2 - p1, p3 - p1)
    z_axis /= np.linalg.norm(z_axis) 

    midpoint = (p1 + p3) / 2
    y_axis = p2 - midpoint
    y_axis /= np.linalg.norm(y_axis) 

    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)  

    rotation_matrix = np.vstack([x_axis, y_axis, z_axis]).T

    return rotation_matrix

rotation_matrix = calculate_local_coordinate_system(base_points_camera)

base_point_robot = np.array([0, 0, 0])
translation_vector = base_point_robot - rotation_matrix.dot(base_points_camera[0])

def transform_target_position(target_position_camera, rotation_matrix, translation_vector):
    target_position_robot = rotation_matrix.dot(target_position_camera) + translation_vector
    return target_position_robot

target_position_robot = transform_target_position(target_position_camera, rotation_matrix, translation_vector)
print("Target Position in Robot Coordinate System:\n", target_position_robot)
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
robot_id = p.loadURDF("/home/rhos/Desktop/LeapTele/DexCopilot/assets/low_cost_robot_description/urdf/low_cost_robot.urdf", useFixedBase=True)
visual_shape_id = p.createVisualShape(
    shapeType=p.GEOM_SPHERE,
    radius=0.01,  
    rgbaColor=[1, 0, 0, 1]  
)

p.createMultiBody(
    baseMass=0,
    baseVisualShapeIndex=visual_shape_id,
    basePosition=target_position_robot
)

for _ in range(10000):
    p.stepSimulation()

# time.sleep(15)
joint_angles = p.calculateInverseKinematics(robot_id, 5, target_position_robot)
print("Joint Angles in Robot Coordinate System:\n", joint_angles)
joint_limits = [
    (-1.5708, 1.5708),  # joint1
    (-0.5236, 1.5708),  # joint2
    (-1.3963, 1.5708),  # joint3
    (0, 3.1416),        # joint4
    (0, 3.1416),        # joint5
    (-1.5708, 0)        # joint_gripper
]

for i in range(len(joint_angles)):
    p.resetJointState(robot_id, i, joint_angles[i])

scaled_joint_angles = []
for i, angle in enumerate(joint_angles):
    lower_limit, upper_limit = joint_limits[i]
    scaled_angle = max(lower_limit, min(upper_limit, angle))
    scaled_joint_angles.append(scaled_angle)
print("Joint Angles in Robot Coordinate System:\n", scaled_joint_angles)
joint_angles_degrees = np.degrees(joint_angles)
print("Joint Angles After Degree:\n", joint_angles_degrees)
joint_angles_real= 90 - joint_angles_degrees
joint_angles_real[0] -= 90
print("Joint Angles After Trans:\n", joint_angles_real)
for _ in range(record_time_s * fps):
    start_time = time.perf_counter()
    follower_pos = robot.follower_arms["main"].read("Present_Position")  
    robot.follower_arms["main"].write("Goal_Position", joint_angles_real)

    # print(follower_pos)

    dt_s = time.perf_counter() - start_time
    busy_wait(1 / fps - dt_s)

robot.disconnect()