# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause


"""
This demo shows how the the multranger deck can be used to measure distances
in particular it show that the same sensor raycasts multiple rays and only the
closest one is kept. 
"""

import os
import sys
import argparse
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Multiranger Deck - Walls Scenario")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import numpy as np
import torch
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import CameraCfg
from isaaclab_assets import CRAZYFLIE_CFG

from source.multiranger_deck_cfg import MultirangerDeckCfg
from scripts.quacopter_control.flight_controller import QuadcopterController

from source.patterns.multiranger_deck_patterns import MultirangerPatternCfg

# Camera Rotations
_rot_matrix = Rotation.from_euler('xyz', [-180, 0, -90], degrees=True)
_x, _y, _z, _w = _rot_matrix.as_quat()
CUSTOM_CAMERA_ROT = (_w, _x, _y, _z)
_x, _y, _z, _w = Rotation.from_euler('xyz', [-160, 0, -80], degrees=True).as_quat() #DRONE POV CAMERA; other angulation [-110, 0, -80]
CUSTOM_CAMERA_ROT2 = (_w, _x, _y, _z)
_rx, _ry, _rz, _rw = Rotation.from_euler('xyz', [0, 0, 0], degrees=True).as_quat()
ROBOT_START_ROT = (_rw, _rx, _ry, _rz)

RAYCAST_TARGETS = [
    "/World/Ground", 
    "/World/WallNorth",
    "/World/WallSouth", 
    "/World/WallEastFirst", "/World/WallWestFirst",
    "/World/WallEastSecond", "/World/WallWestSecond"
]

@configclass
class WallsSceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot."""
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Grid/default_environment.usd")
    )

    # FOUR WALLS TO TRAP THE DRONE
    wall_north = AssetBaseCfg(
        prim_path="/World/WallNorth",
        spawn=sim_utils.CuboidCfg(size=(0.1, 2.0, 3.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.7))), 
        init_state=AssetBaseCfg.InitialStateCfg(pos=(3.0, 0.0, 1.0)) 
    )
    
    wall_south = AssetBaseCfg(
        prim_path="/World/WallSouth",
        spawn=sim_utils.CuboidCfg(size=(0.1, 4.0, 3.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.7))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-1.0, 0.0, 1.0)) 
    )
    
    wall_east_first = AssetBaseCfg(
        prim_path="/World/WallEastFirst",
        spawn=sim_utils.CuboidCfg(size=(2.0, 0.1, 3.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.7))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 2.0, 1.0)) 
    )
    wall_east_second = AssetBaseCfg(
        prim_path="/World/WallEastSecond",
        spawn=sim_utils.CuboidCfg(size=(2.0, 0.1, 3.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.7))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(2.0, 1.0, 1.0)) 
    )

    wall_west_first = AssetBaseCfg(
        prim_path="/World/WallWestFirst",
        spawn=sim_utils.CuboidCfg(size=(2.0, 0.1, 3.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.7))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -2.0, 1.0))
    )
    wall_west_second = AssetBaseCfg(
        prim_path="/World/WallWestSecond",
        spawn=sim_utils.CuboidCfg(size=(2.0, 0.1, 3.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.7))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(2.0, -1.0, 1.0))
    )

    robot = CRAZYFLIE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos = (0.0, 0.0, 0.15) #start position at 15cm from GND
    robot.init_state.rot = ROBOT_START_ROT
    
    multiranger = MultirangerDeckCfg(
        prim_path="{ENV_REGEX_NS}/Robot", 
        update_period=1 / 60,
        offset=MultirangerDeckCfg.OffsetCfg(pos=(0, 0, 0.15)),
        mesh_prim_paths=RAYCAST_TARGETS,
        ray_alignment="yaw",
        max_distance=4.0, 
        debug_vis=True,
        pattern_cfg=MultirangerPatternCfg(
            fov_degrees=15.0,
            #rays_per_cone=1
        ),
    )
    
    camera = CameraCfg(     #Camera from above
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=1 / 30,  
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(1.0, 0.0, 8.0),
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
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    robot = scene["robot"]
    prop_body_ids = robot.find_bodies("m.*_prop")[0]
    
    # MISSION PARAMETERS
    drone_controller = QuadcopterController(
        target_height=1.0,  
        cruise_vel=0.5,
        sim=sim,
        debug=True
    )

    front_props, rear_props = [0, 3], [1, 2]
    sim.reset()
    
    save_dir = os.path.join(parent_dir, "multimedia", "demo2")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # VIDEO
    video_path = os.path.join(save_dir, "flight.mp4")
    print(f"[INFO] Start recording video: {video_path}")
    video_writer = imageio.get_writer(video_path, fps=30)

    # REGISTERING
    log_time = []
    log_front, log_back, log_left, log_right, log_down = [], [], [], [], []
    img_np_pov = None
    img_np_top = None

    #phase = 1               # 1: climb: 2: movement forward
    #target_cruise_z = 1.5   # final altitude
    #ascent_rate = 0.003     # climbing velocity
    
    print("\n[INFO] Start moving")

    while simulation_app.is_running():
        if count > 1300.0:
            break 
            
        ranges = scene["multiranger"].data.ranges  
        front_range = float(ranges[0, 0].item())  
        back_range  = float(ranges[0, 1].item())
        left_range  = float(ranges[0, 2].item())
        right_range = float(ranges[0, 3].item())
        down_range  = float(ranges[0, 4].item()) 

        if front_range < 0.3:
            drone_controller.set_cruise_velocity(0.0)
            
            
        # if phase == 1:
        #     if drone_controller.target_height < target_cruise_z:
                
        #         drone_controller.target_height += ascent_rate
        #     else:
        #         print("[INFO] Phase 2: Start moving forward")
        #         phase = 2
        #         drone_controller.cruise_pitch = 0.07

        # elif phase == 2:
            
        #     pass

        # --- LOG MULTIRANGER DATA ---
        log_time.append(sim_time)
        log_front.append(front_range)
        log_back.append(back_range)
        log_left.append(left_range)
        log_right.append(right_range)
        log_down.append(down_range)
        
        # Flight Controller logic
        root_quat = robot.data.root_quat_w
        ang_vel = robot.data.root_ang_vel_w 
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(root_quat)
        
        current_pitch = pitch[0].item()
        pitch_rate = ang_vel[0, 1].item()
        vx = float(robot.data.body_com_vel_w[0, 0][0].item())
        vz = float(robot.data.root_lin_vel_w[0, 2].item())
        ax = float(robot.data.body_lin_acc_w[0, 0][0].item())

        robot_mass = float(robot.root_physx_view.get_masses().sum().item())
        gravity = torch.tensor(sim.cfg.gravity, device=sim.device).norm().item()
        hover_per = (robot_mass * gravity) / 4.0

        front_thrust, rear_thrust = drone_controller.update(
            down_range, current_pitch, pitch_rate, vx, vz, ax, hover_per
        )
        
        forces = torch.zeros(robot.num_instances, 4, 3, device=sim.device)
        forces[:, front_props, 2] = front_thrust
        forces[:, rear_props,  2] = rear_thrust
        
        robot.permanent_wrench_composer.set_forces_and_torques(
            forces=forces, torques=torch.zeros_like(forces), body_ids=prop_body_ids
        )

        scene.write_data_to_sim()
        sim.step(render=True)
        sim_time += sim_dt
        scene.update(sim_dt)

        # Videoo
        top_rgb_tensor = scene["camera"].data.output["rgb"]
        pov_rgb_tensor = scene["camera2"].data.output["rgb"]
        
        if pov_rgb_tensor is not None and top_rgb_tensor is not None:
            frame_top = top_rgb_tensor[0].clone().cpu().numpy()
            frame_pov = pov_rgb_tensor[0].clone().cpu().numpy()

            if frame_top.shape[-1] == 4: 
                frame_top = frame_top[..., :3]
            if frame_pov.shape[-1] == 4: 
                frame_pov = frame_pov[..., :3]

            img_np_top = frame_top
            img_np_pov = frame_pov

            if count % 6 == 0: #1 frame per 6 step: 200Hz / 30FPS = 6.6666...
                video_writer.append_data(frame_pov.astype(np.uint8))
        
        count += 1

    print(f"[INFO] Saving video...")
    video_writer.close()

    images = scene["camera"].data.output["rgb"]
    print("[INFO] Saving final camera snapshots...")
    if img_np_top is not None:
        Image.fromarray(img_np_top.astype(np.uint8)).save(f"{save_dir}/wall_distance_demo.png")

        
    print("[INFO] Generating Multiranger plot...")
    if len(log_time) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        ax.plot(log_time, log_front, color="blue", label="Front Range")
        ax.plot(log_time, log_back, color="orange", label="Back Range")
        ax.plot(log_time, log_left, color="green", label="Left Range")
        ax.plot(log_time, log_right, color="red", label="Right Range")
        ax.plot(log_time, log_down, color="purple", linestyle="--", label="Down Range (Altitude)")
        
        ax.set_title("Multiranger Sensor Data (Wall Hover Scenario)")
        ax.set_ylabel("Distance (meters)")
        ax.set_xlabel("Simulation Time (seconds)")
        ax.set_ylim(0, 5)
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/wall_distance_demo_plt.png")
        print(f"[INFO] Saved {save_dir}/wall_distance_demo_plt.png")

    drone_controller.plot_debug(save_dir)
def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])
    scene_cfg = WallsSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    run_simulator(sim, scene)

if __name__ == "__main__":
    main()
    simulation_app.close()