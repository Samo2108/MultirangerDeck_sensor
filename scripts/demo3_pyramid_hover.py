# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
This demo shows how to use the Multiranger Deck sensor to perform
a simple wall-following behavior.
In this scenario the drone takes off and tries to mantain a certain
distance from ground and nearby walls.
"""
import argparse
from isaaclab.app import AppLauncher
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

# add argparse arguments
parser = argparse.ArgumentParser(description="Example on using the custom Multiranger Deck sensor.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import numpy as np
import torch
from dataclasses import dataclass

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import CameraCfg
from PIL import Image

# Import base Raycaster components
from source.multiranger_deck_cfg import MultirangerDeckCfg

##
# Pre-defined configs
##
from scipy.spatial.transform import Rotation
import imageio
from isaaclab_assets import CRAZYFLIE_CFG  # isort:skip
import isaaclab.utils.math as math_utils
import matplotlib
matplotlib.use("Agg")  # must be before importing pyplot
import matplotlib.pyplot as plt
from scripts.quacopter_control.flight_controller import QuadcopterController

_rot_matrix = Rotation.from_euler('xyz', [-180, -0, 0], degrees=True)
_x, _y, _z, _w = _rot_matrix.as_quat()
CUSTOM_CAMERA_ROT = (_w, _x, _y, _z)
_rot_matrix = Rotation.from_euler('xyz', [-90, -0, 0], degrees=True)
_x, _y, _z, _w = _rot_matrix.as_quat()
CUSTOM_CAMERA_ROT2 = (_w, _x, _y, _z)
_rot_matrix = Rotation.from_euler('XYZ', [-0, 105, -90], degrees=True)
_x, _y, _z, _w = _rot_matrix.as_quat()
CUSTOM_CAMERA_ROT3 = (_w, _x, _y, _z)


@configclass
class RaycasterSensorSceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot."""

    # GROUND
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Grid/default_environment.usd",
        ),
    )
    
    # THE PYRAMID
    first_level = AssetBaseCfg(
        prim_path="/World/FirstLevel",
        spawn=sim_utils.CuboidCfg(size=(5.0, 5.0, 0.3), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.7))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.1)) 
    )
    second_level = AssetBaseCfg(
        prim_path="/World/SecondLevel",
        spawn=sim_utils.CuboidCfg(size=(3.0, 3.0, 0.5), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.7))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.3)) 
    )
    third_level = AssetBaseCfg(
        prim_path="/World/ThirdLevel",
        spawn=sim_utils.CuboidCfg(size=(1.0, 1.0, 0.5), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.7))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)) 
    )
    first_level1 = AssetBaseCfg(
        prim_path="/World/FirstLevel1",
        spawn=sim_utils.CuboidCfg(size=(5.0, 5.0, 0.3), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.7))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(6.0, 0.0, 0.1)) 
    )
    second_level1 = AssetBaseCfg(
        prim_path="/World/SecondLevel1",
        spawn=sim_utils.CuboidCfg(size=(3.0, 3.0, 0.5), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.7))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(6.0, 0.0, 0.3)) 
    )
    third_level1 = AssetBaseCfg(
        prim_path="/World/ThirdLevel1",
        spawn=sim_utils.CuboidCfg(size=(1.0, 1.0, 0.5), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.7))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(6.0, 0.0, 0.5)) 
    )
    contrast_wall = AssetBaseCfg(
        prim_path="/World/ContrastWall",
        spawn=sim_utils.CuboidCfg(size=(20.0, 0.1, 10.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.7, 0.1))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 6.0, 5.0))
    )

    # MOCK DRONE
    robot = CRAZYFLIE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    robot.init_state.pos = (0.0, 0.0, 1.3) 
    
    # MULTIRANGER
    multiranger = MultirangerDeckCfg(
        prim_path="{ENV_REGEX_NS}/Robot", 
        update_period=1 / 60,
        offset=MultirangerDeckCfg.OffsetCfg(pos=(0, 0, 0)),
        mesh_prim_paths=[
        "/World/Ground",
        "/World/FirstLevel",
        "/World/SecondLevel",
        "/World/ThirdLevel",
        "/World/FirstLevel1",
        "/World/SecondLevel1",
        "/World/ThirdLevel1"
        ],  
        ray_alignment="yaw",
        max_distance=4.0, 
        debug_vis=True, 
    )
    
    # SATELLITE CAMERA
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=1 / 30,  
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(3.0, -0.0, 8.0),
            rot=CUSTOM_CAMERA_ROT
        ),
    )
    
    # SIDE CAMERA
    camera2 = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera2",
        update_period=1 / 30,  
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(3.0, -7.0, 1.0),
            rot=CUSTOM_CAMERA_ROT2
        ),
    )
    
    # DRONE POVE CAMERA
    camera3 = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body/Camera3",
        update_period=1 / 30,  
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(-3, 0.0, 0.5),
            rot=CUSTOM_CAMERA_ROT3
        ),
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator."""
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    robot = scene["robot"]
   
    prop_body_ids = robot.find_bodies("m.*_prop")[0]
    
    # ========================================================
    # MISSION PARAMETERS (Set these to whatever you want!)
    # ========================================================
    target_height = 0.3     # meters
    stop_distance = 0.0     # meters
    cruise_pitch = 0.1      # radians (~5.7 degrees)

    # Initialize the separated brain
    drone_controller = QuadcopterController(
        target_height=target_height, 
        stop_distance=stop_distance, 
        cruise_pitch=cruise_pitch
    )
    # ========================================================

    front_props = [0, 3]    
    rear_props  = [1, 2]
    
    sim.reset()
    
    if not os.path.exists("MultirangerDeck/multimedia/demo3/"):
        os.makedirs("MultirangerDeck/multimedia/demo3/")

    video_writer = imageio.get_writer("MultirangerDeck/multimedia/demo3/pyramid_hover_demo.mp4", fps=30)
    video_writer2 = imageio.get_writer("MultirangerDeck/multimedia/demo3/pyramid_hover_demo_cam2.mp4", fps=30)
    video_writer3 = imageio.get_writer("MultirangerDeck/multimedia/demo3/pyramid_hover_demo_cam3.mp4", fps=30)
    
    log_time, log_distance, log_target_pitch = [], [], []
    log_actual_pitch, log_thrust_diff = [], []
    log_actual_height, log_mesured_height = [], []
    
    sim_limit = 1000.0  # seconds
    print(f"[INFO] Simulating for {sim_limit} steps...")
    while simulation_app.is_running():
        if count > sim_limit:
            break
        
        # SENSOR READINGS 
        ranges = scene["multiranger"].data.ranges  
        front_range = float(ranges[0, 0].item())  
        down_range  = float(ranges[0, 4].item()) 
        
        root_quat = robot.data.root_quat_w
        ang_vel = robot.data.root_ang_vel_w 
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(root_quat)
        
        current_pitch = pitch[0].item()
        pitch_rate = ang_vel[0, 1].item()
        vx = float(robot.data.root_lin_vel_w[0, 0].item())
        vz = float(robot.data.root_lin_vel_w[0, 2].item())

        robot_mass = float(robot.root_physx_view.get_masses().sum().item())
        gravity = torch.tensor(sim.cfg.gravity, device=sim.device).norm().item()
        hover_per = (robot_mass * gravity) / 4.0

        # UPDATE DRONE COMANDS
        front_thrust, rear_thrust, target_pitch, pitch_command = drone_controller.update(
            front_range, down_range, 
            current_pitch, pitch_rate, 
            vx, vz, hover_per
        )

        # ---  LOG DATA ---
        log_time.append(sim_time)
        log_distance.append(front_range)
        log_target_pitch.append(math.degrees(target_pitch)) 
        log_actual_pitch.append(math.degrees(current_pitch))
        log_thrust_diff.append(pitch_command)
        log_actual_height.append(robot.data.root_pos_w[0, 2].item())
        log_mesured_height.append(down_range)
        
        # APPLY FORCES
        forces = torch.zeros(robot.num_instances, 4, 3, device=sim.device)
        torques = torch.zeros_like(forces)

        forces[:, front_props, 2] = front_thrust
        forces[:, rear_props,  2] = rear_thrust
        
        robot.permanent_wrench_composer.set_forces_and_torques(
            forces=forces, torques=torques, body_ids=prop_body_ids
        )

    
        scene.write_data_to_sim()
        sim.step(render=True)
        sim_time += sim_dt
        scene.update(sim_dt)

        # RECORD VIDEO 
        img_tensor = scene["camera"].data.output["rgb"][0]
        img_np = img_tensor.cpu().numpy()
        img_tensor2 = scene["camera2"].data.output["rgb"][0]
        img_np2 = img_tensor2.cpu().numpy()
        img_tensor3 = scene["camera3"].data.output["rgb"][0]
        img_np3 = img_tensor3.cpu().numpy()
        
        if img_np.shape[-1] == 4: img_np = img_np[..., :3]
        if img_np.dtype != np.uint8: img_np = img_np.astype(np.uint8)
        if img_np2.shape[-1] == 4: img_np2 = img_np2[..., :3]
        if img_np2.dtype != np.uint8: img_np2 = img_np2.astype(np.uint8)
        if img_np3.shape[-1] == 4: img_np3 = img_np3[..., :3]
        if img_np3.dtype != np.uint8: img_np3 = img_np3.astype(np.uint8)
            
        video_writer.append_data(img_np)
        video_writer2.append_data(img_np2)
        video_writer3.append_data(img_np3)
        
        count += 1


    print("[INFO] Closing videos and generating plots...")
    video_writer.close()
    video_writer2.close()
    video_writer3.close()

    if len(log_time) > 0:
        fig, axs = plt.subplots(3, 2, figsize=(10, 12), sharex=True)
        
        # Plot 1: Distance to Wall
        axs[0, 0].plot(log_time, log_distance, label="Front Laser Range (m)", color="blue")
        axs[0, 0].axhline(y=stop_distance, color='red', linestyle='--', label="Target Stop Distance")
        axs[0, 0].set_ylabel("Distance (meters)")
        axs[0, 0].set_title("Wall Approach Distance")
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        # Plot 2: Pitch Tracking
        axs[1, 0].plot(log_time, log_target_pitch, label="Target Pitch (deg)", linestyle="--", color="orange")
        axs[1, 0].plot(log_time, log_actual_pitch, label="Actual Pitch (deg)", color="green")
        axs[1, 0].set_ylabel("Angle (degrees)")
        axs[1, 0].set_title("Pitch Controller Tracking")
        axs[1, 0].legend()
        axs[1, 0].grid(True)

        # Plot 3: Motor Effort
        axs[2, 0].plot(log_time, log_thrust_diff, label="Differential Thrust Cmd (N)", color="purple")
        axs[2, 0].set_ylabel("Force (Newtons)")
        axs[2, 0].set_xlabel("Simulation Time (seconds)")
        axs[2, 0].set_title("Motor Effort")
        axs[2, 0].legend()
        axs[2, 0].grid(True)

        # Plot 4: Height Tracking
        axs[0, 1].plot(log_time, log_actual_height, label="Actual Height (m)", color="blue")
        axs[0, 1].plot(log_time, log_mesured_height, label="Measured Height (m)", color="green")
        axs[0, 1].set_ylabel("Height (meters)")
        axs[0, 1].set_title("Height Tracking")
        axs[0, 1].legend()
        axs[0, 1].grid(True)

        plt.tight_layout()
        plt.savefig("MultirangerDeck/multimedia/demo3/pyramid_hover_telemetry.png")
        print("[INFO] Plot saved to pyramid_hover_telemetry.png!")

def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])
    
    scene_cfg = RaycasterSensorSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()