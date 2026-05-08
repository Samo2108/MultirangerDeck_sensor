# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys
import argparse
import torch
import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description="Demo: Hardware-Truth Point Cloud Mapping")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils

from isaaclab.sensors import CameraCfg
from isaaclab_assets import CRAZYFLIE_CFG
from source.multiranger_deck_cfg import MultirangerDeckCfg
from scripts.quacopter_control.flight_controller import QuadcopterController

from source.patterns.multiranger_deck_patterns import MultirangerPatternCfg
import imageio
from scipy.spatial.transform import Rotation
# Filter threshold for the point cloud (meters)
# Anything beyond this is ignored, matching the Bitcraze demo logic
SENSOR_TH = 2.0  


_x, _y, _z, _w = Rotation.from_euler('XYZ', [-180, 0, 0], degrees=True).as_quat()
CUSTOM_CAMERA_ROT = (_w, _x, _y, _z)


@configclass
class PointCloudSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Grid/default_environment.usd")
    )
    
    corridor1_wall_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Corr1WallLeft",
        spawn=sim_utils.CuboidCfg(size=(1.6, 0.02, 1.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.7))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.8, 0.5, 0.5)) 
    )
    
    corridor1_wall_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Corr1WallRight",
        spawn=sim_utils.CuboidCfg(size=(0.5, 0.02, 1.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.7))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.25, -0.5, 0.5)) 
    )

    center_sphere = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CenterSphere",
        spawn=sim_utils.CylinderCfg(radius=0.3, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.3, 0.3))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.8, -0.5, 0.5)) 
    )
    
    corridor2_wall_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Corr2WallRight",
        spawn=sim_utils.CuboidCfg(size=(0.5, 0.02, 1.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.7))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.35, -0.5, 0.5)) 
    )
    
    end_wall = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/EndWall",
        spawn=sim_utils.CuboidCfg(size=(0.02, 1.0, 1.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.7))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.6, 0.0, 0.5)) 
    )

    robot = CRAZYFLIE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos = (0.0, 0.0, 0.3) 
    
    
    multiranger = MultirangerDeckCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body", 
        update_period=1 / 60,
        mesh_prim_paths=["/World/Ground",
                         "{ENV_REGEX_NS}/Corr1WallLeft",
                         "{ENV_REGEX_NS}/Corr1WallRight",
                         "{ENV_REGEX_NS}/Corr2WallRight",
                         "{ENV_REGEX_NS}/CenterSphere",
                         "{ENV_REGEX_NS}/EndWall"],  
        #ray_alignment="yaw",
        max_distance=4.0, 
        debug_vis=True,
        pattern_cfg=MultirangerPatternCfg(
            fov_degrees=15.0,
        ),
    )
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/camera",
        update_period=1 / 30,  height=480, width=640, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955),
        offset=CameraCfg.OffsetCfg(pos=(0.8, 0.0, 3.0), rot=CUSTOM_CAMERA_ROT),
    )

def run_pointcloud_mapper(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    robot = scene["robot"]
    prop_body_ids = robot.find_bodies("m.*_prop")[0]
    
    save_dir = os.path.join(parent_dir, "multimedia", "demo3")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    drone_controller = QuadcopterController(target_height=0.3, target_vel=0.001, sim=sim, debug=True)
    video_writer = imageio.get_writer(os.path.join(save_dir, "demo3.mp4"), fps=30)
    # POINT CLOUD DATA STORAGE
    point_cloud_w = [] 

    print("\n[INFO] Starting Hardware-Truth Point Cloud Mapping...")
    nav_state = "FLY"
    for step in range(7000):
        if not simulation_app.is_running(): break
        
        # READ THE 1D SENSOR RANGES 
        ranges = scene["multiranger"].data.ranges[0]
        root_pose = robot.data.root_pos_w[0]
        root_quat = robot.data.root_quat_w[0]

        # Define the local 1D laser vectors (Front, Back, Left, Right)
        # down is left out
        ray_dirs_local = torch.tensor([
            [1.0, 0.0, 0.0],   
            [-1.0, 0.0, 0.0],  
            [0.0, 1.0, 0.0],   
            [0.0, -1.0, 0.0]   
        ], device=sim.device)

        # Apply the drone's current rotation (Quaternion) to the local laser vectors
        ray_dirs_world = math_utils.quat_apply(root_quat.expand(4, 4), ray_dirs_local)

        # Filter out max distances and project the 3D hit point
        for i in range(4):
            dist = ranges[i].item()
            # If the measurement is less than our threshold, map it!
            if dist < SENSOR_TH:
                # Drone Position + (World Direction Vector * Distance)
                hit_pos = root_pose + (ray_dirs_world[i] * dist)
                point_cloud_w.append([hit_pos[0].item(), hit_pos[1].item(), hit_pos[2].item()])


        vx = float(robot.data.root_lin_vel_b[0, 0].item()) 
        vy = float(robot.data.root_lin_vel_b[0, 1].item()) 
        vz = float(robot.data.root_lin_vel_w[0, 2].item()) 
        ax = float(robot.data.body_lin_acc_w[0, 0][0].item())
        ay = float(robot.data.body_lin_acc_w[0, 1][1].item())
        
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(robot.data.root_quat_w)
        current_roll  = roll[0].item()
        current_pitch = pitch[0].item()
        current_yaw   = yaw[0].item()
        
        # Body angular rates
        roll_rate  = float(robot.data.root_ang_vel_b[0, 0].item())
        pitch_rate = float(robot.data.root_ang_vel_b[0, 1].item())
        yaw_rate   = float(robot.data.root_ang_vel_b[0, 2].item())
        
        # World Yaw
        current_yaw = math_utils.euler_xyz_from_quat(robot.data.root_quat_w)[2][0].item()

        if nav_state == "FLY":
            drone_controller.set_cruise_velocity(0.1)
            drone_controller.set_yaw(None)  
            
            if root_pose[0].item() >= 1.4:
                nav_state = "BRAKE"
                print("[INFO] Target reached...")

        elif nav_state == "BRAKE":
            drone_controller.set_cruise_velocity(0.0)
            drone_controller.set_yaw(None)  
            
            if abs(vx) < 0.1:  
                nav_state = "SPIN"
                print("[INFO] Spinning")

        elif nav_state == "SPIN":
            drone_controller.set_cruise_velocity(0.0)
            target = current_yaw + 0.005 
            drone_controller.set_yaw(target)

        # flight controller
        robot_mass = float(robot.root_physx_view.get_masses().sum().item())
        gravity = torch.tensor(sim.cfg.gravity, device=sim.device).norm().item()
        hover_thrust = (robot_mass * gravity) / 4.0
        
        m0, m1, m2, m3 = drone_controller.update(
            down_range=ranges[4].item(),
            current_pitch=current_pitch,
            pitch_rate=pitch_rate,
            current_roll=current_roll,
            roll_rate=roll_rate,
            vx=vx,
            vy=vy,
            vz=vz,
            ax=ax,
            ay=ay,
            base_hover_thrust=hover_thrust,
            current_yaw=current_yaw,
            yaw_rate=yaw_rate
        )

        # Apply them to the 4 specific propeller bodies in Isaac Lab
        forces = torch.zeros(robot.num_instances, 4, 3, device=sim.device)
        forces[:, 0, 2] = m0  # Front-Right
        forces[:, 1, 2] = m1  # Rear-Right
        forces[:, 2, 2] = m2  # Rear-Left
        forces[:, 3, 2] = m3  # Front-Left
        
        # THE DRAG SIMULATOR 
        c_drag = 3.0
        torques = torch.zeros_like(forces)
        torques[:, 0, 2] = -m0 * c_drag  # CW torque
        torques[:, 1, 2] =  m1 * c_drag  # CCW torque
        torques[:, 2, 2] = -m2 * c_drag  # CW torque
        torques[:, 3, 2] =  m3 * c_drag  # CCW torque
        
        # SEND ONCE
        robot.permanent_wrench_composer.set_forces_and_torques(
            forces=forces, 
            torques=torques, 
            body_ids=prop_body_ids
        )

        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_dt)
        
        img_tensor = scene["camera"].data.output["rgb"][0]
        img_np = img_tensor.cpu().numpy()
        
        if img_np.shape[-1] == 4: img_np = img_np[..., :3]
        if img_np.dtype != np.uint8: img_np = img_np.astype(np.uint8)
        if step % 6 == 0: #1 frame per 6 step: 200Hz / 30FPS = 6.6666...    
            video_writer.append_data(img_np)
        
    drone_controller.plot_debug(save_dir)
    video_writer.close()
    print(f"[INFO] Mapping complete. Captured {len(point_cloud_w)} points. Generating 2D Plot...")
    
    fig = plt.figure(figsize=(10, 8))
    # 2D plot
    ax = fig.add_subplot(111) 
    
    pc = np.array(point_cloud_w)
    
    if len(pc) > 0:
        # Scatter only X and Y
        ax.scatter(pc[:, 0], pc[:, 1], color='blue', s=10, alpha=0.6, label="Sensor Hits")
        
    ax.set_title("2D Multiranger Map (Top-Down View)")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    
    if len(pc) > 0:
        ax.set_aspect('equal', adjustable='box')
    
    plt.legend()
    plt.grid(True) # Added a grid to make the distances easier to read
    
    save_path = os.path.join(save_dir, "demo_pointcloud.png")
    
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
        
    plt.savefig(save_path, dpi=300)
    print(f"[INFO] 2D Point cloud saved to: {save_path}")

def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[-2.0, 3.0, 4.0], target=[3.0, 0.0, 0.5])
    
    scene_cfg = PointCloudSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    sim.reset() 
    run_pointcloud_mapper(sim, scene)

if __name__ == "__main__":
    main()
    simulation_app.close()