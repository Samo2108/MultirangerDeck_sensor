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

_rot_matrix = Rotation.from_euler('XYZ', [-105, -0, 0], degrees=True)
_x, _y, _z, _w = _rot_matrix.as_quat()
CUSTOM_CAMERA_ROT = (_w, _x, _y, _z)

_x, _y, _z, _w = Rotation.from_euler('xyz', [-160, 0, -80], degrees=True).as_quat() #DRONE POV CAMERA; other angulation [-110, 0, -80]
CUSTOM_CAMERA_ROT2 = (_w, _x, _y, _z)

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
        spawn=sim_utils.CuboidCfg(size=(5.0, 2.0, 0.3), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.7))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.15)) 
    )
    second_level = AssetBaseCfg(
        prim_path="/World/SecondLevel",
        spawn=sim_utils.CuboidCfg(size=(3.0, 2.0, 0.3), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.7))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.45)) 
    )
    third_level = AssetBaseCfg(
        prim_path="/World/ThirdLevel",
        spawn=sim_utils.CuboidCfg(size=(1.0, 2.0, 0.3), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.7))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.75)) 
    )
    first_level1 = AssetBaseCfg(
        prim_path="/World/FirstLevel1",
        spawn=sim_utils.CuboidCfg(size=(5.0, 2.0, 0.3), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.7))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(6.0, 0.0, 0.15)) 
    )
    second_level1 = AssetBaseCfg(
        prim_path="/World/SecondLevel1",
        spawn=sim_utils.CuboidCfg(size=(3.0, 2.0, 0.3), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.7))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(6.0, 0.0, 0.45)) 
    )
    third_level1 = AssetBaseCfg(
        prim_path="/World/ThirdLevel1",
        spawn=sim_utils.CuboidCfg(size=(1.0, 2.0, 0.3), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.7))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(6.0, 0.0, 0.75)) 
    )
    contrast_wall = AssetBaseCfg(
        prim_path="/World/ContrastWall",
        spawn=sim_utils.CuboidCfg(size=(20.0, 0.1, 10.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.7, 0.1))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 1.05, 5.0))
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
    
    # SIDE CAMERA
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera2",
        update_period=1 / 30,  
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(3.0, -6.0, 2.0),
            rot=CUSTOM_CAMERA_ROT
        ),
    )
    camera2 = CameraCfg(        #Camera with drone POV
        prim_path="{ENV_REGEX_NS}/Robot/body/Camera3",
        update_period=1 / 30,  
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(-1.0, -0.25, 3.0),rot=CUSTOM_CAMERA_ROT2),
            #pos=(-0.8, -0.15, 0.3),rot=CUSTOM_CAMERA_ROT2), #other angulation
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator."""
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    robot = scene["robot"]
   
    prop_body_ids = robot.find_bodies("m.*_prop")[0]

    # MISSION PARAMETERS 
    target_height = 0.4    # meters
    cruise_vel = 1.5          # m/s
    
    # Initialize controller
    drone_controller = QuadcopterController(
        target_height=target_height, 
        target_vel=cruise_vel,
        sim=sim,
        debug=True
    )

    front_props = [0, 3]    
    rear_props  = [1, 2]
    
    sim.reset()
    
    save_dir = os.path.join(parent_dir, "multimedia", "demo3")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    video_writer = imageio.get_writer(os.path.join(save_dir, "pyramid_hover_demo.mp4"), fps=30)
    video_writer2 = imageio.get_writer(os.path.join(save_dir, "pyramid_hover_demo2.mp4"), fps=30)
    
    log_time = []
    log_actual_height = []
    log_mesured_height = []
    
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
        ax = float(robot.data.body_lin_acc_w[0, 0][0].item())
        
        robot_mass = float(robot.root_physx_view.get_masses().sum().item())
        gravity = torch.tensor(sim.cfg.gravity, device=sim.device).norm().item()
        hover_per = (robot_mass * gravity) / 4.0

        # UPDATE DRONE COMANDS
        front_thrust, rear_thrust = drone_controller.update(
            down_range, 
            current_pitch, pitch_rate, 
            vx, vz, ax, hover_per
        )

        # ---  LOG DATA ---
        log_time.append(sim_time)
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
        
        if img_np.shape[-1] == 4: img_np = img_np[..., :3]
        if img_np.dtype != np.uint8: img_np = img_np.astype(np.uint8)
        
        # RECORD VIDEO FROM DRONE POV
        img_tensor2 = scene["camera2"].data.output["rgb"][0]
        img_np2 = img_tensor2.cpu().numpy()
        
        if img_np2.shape[-1] == 4:
            img_np2 = img_np2[..., :3]
        if img_np2.dtype != np.uint8:
            img_np2 = img_np2.astype(np.uint8)

        if count % 6 == 0: #1 frame per 6 step: 200Hz / 30FPS = 6.6666...    
            video_writer2.append_data(img_np2)
            video_writer.append_data(img_np)
        
        count += 1


    print("[INFO] Closing videos and generating plots...")
    video_writer.close()
    video_writer2.close()

    if len(log_time) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(log_time, log_actual_height, label="Actual Drone Height (m)", color="blue")
        plt.plot(log_time, log_mesured_height, label="Multiranger Measured Height (m)", color="green")
        
        plt.ylabel("Height (meters)")
        plt.xlabel("Simulation Time (seconds)")
        plt.title("Mission Output")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "pyramid_surfing.png"))
        plt.close()
        print("[INFO] Mission plot saved to pyramid_surfing.png!")

    drone_controller.plot_debug(save_dir)


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