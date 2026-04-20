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

# Camera Rotations
_x, _y, _z, _w = Rotation.from_euler('xyz', [-180, -0, 0], degrees=True).as_quat()
CUSTOM_CAMERA_ROT = (_w, _x, _y, _z)

RAYCAST_TARGETS = [
    "/World/Ground", 
    "/World/WallNorth_Right", "/World/WallNorth_Left",
    "/World/WallSouth_Right", "/World/WallSouth_Left", 
    "/World/WallEast", "/World/WallWest"
]

@configclass
class WallsSceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot."""
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Grid/default_environment.usd")
    )

    # FOUR WALLS TO TRAP THE DRONE
    wall_north_right = AssetBaseCfg(
        prim_path="/World/WallNorth_Right",
        spawn=sim_utils.CuboidCfg(size=(0.1, 2.0, 2.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.7))), 
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.0, 1.0, 1.0)) 
    )
    wall_north_left = AssetBaseCfg(
        prim_path="/World/WallNorth_Left",
        spawn=sim_utils.CuboidCfg(size=(0.1, 2.0, 2.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.7))), 
        init_state=AssetBaseCfg.InitialStateCfg(pos=(3.0, -1.0, 1.0)) 
    )
    wall_south_right = AssetBaseCfg(
        prim_path="/World/WallSouth_Right",
        spawn=sim_utils.CuboidCfg(size=(0.1, 2.0, 2.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.7))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-1.0, 1.0, 1.0)) 
    )
    wall_south_left = AssetBaseCfg(
        prim_path="/World/WallSouth_Left",
        spawn=sim_utils.CuboidCfg(size=(0.1, 2.0, 2.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.7))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-3.0, -1.0, 1.0)) 
    )
    wall_east = AssetBaseCfg(
        prim_path="/World/WallEast",
        spawn=sim_utils.CuboidCfg(size=(4.0, 0.1, 2.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.7))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 2.0, 1.0)) 
    )
    wall_west = AssetBaseCfg(
        prim_path="/World/WallWest",
        spawn=sim_utils.CuboidCfg(size=(4.0, 0.1, 2.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.7))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -2.0, 1.0))
    )

    robot = CRAZYFLIE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos = (0.0, 0.0, 1.0) 
    
    multiranger = MultirangerDeckCfg(
        prim_path="{ENV_REGEX_NS}/Robot", 
        update_period=1 / 60,
        offset=MultirangerDeckCfg.OffsetCfg(pos=(0, 0, 0)),
        mesh_prim_paths=RAYCAST_TARGETS,
        ray_alignment="yaw",
        max_distance=4.0, 
        debug_vis=True, 
    )
    
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=1 / 30,  height=480, width=640, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 8.0), rot=CUSTOM_CAMERA_ROT),
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    robot = scene["robot"]
    prop_body_ids = robot.find_bodies("m.*_prop")[0]
    
    # MISSION PARAMETERS
    drone_brain = QuadcopterController(
        target_height=1.0,   # Hover securely inside the walls
        stop_distance=1.0,   
        cruise_pitch=0.0     # Hover in place (0.0 means don't move forward)
    )

    front_props, rear_props = [0, 3], [1, 2]
    sim.reset()
    
    if not os.path.exists("MultirangerDeck/multimedia/demo2/"):
        os.makedirs("MultirangerDeck/multimedia/demo2/")

    # REGISTERING
    log_time = []
    log_front, log_back, log_left, log_right, log_down = [], [], [], [], []
    img_np = img_np2 = None
    
    
    while simulation_app.is_running():
        if count > 500.0:
            break # Run for a shorter time since it's just hovering
            
        ranges = scene["multiranger"].data.ranges  
        front_range = float(ranges[0, 0].item())  
        back_range  = float(ranges[0, 1].item())
        left_range  = float(ranges[0, 2].item())
        right_range = float(ranges[0, 3].item())
        down_range  = float(ranges[0, 4].item()) 
        
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
        vx = float(robot.data.root_lin_vel_w[0, 0].item())
        vz = float(robot.data.root_lin_vel_w[0, 2].item())

        robot_mass = float(robot.root_physx_view.get_masses().sum().item())
        gravity = torch.tensor(sim.cfg.gravity, device=sim.device).norm().item()
        hover_per = (robot_mass * gravity) / 4.0

        front_thrust, rear_thrust, _, _ = drone_brain.update(
            front_range, down_range, current_pitch, pitch_rate, vx, vz, hover_per
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

        # Keep overwriting the image variables so we always have the latest frame
        img_np = scene["camera"].data.output["rgb"][0].cpu().numpy()
        if img_np.shape[-1] == 4: img_np = img_np[..., :3]
        
        count += 1


    print("[INFO] Saving final camera snapshots...")
    if img_np is not None:
        Image.fromarray(img_np.astype(np.uint8)).save("MultirangerDeck/multimedia/demo2/wall_distance_demo.png")

        
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
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig("MultirangerDeck/multimedia/demo2/wall_distance_demo_plt.png")
        print("[INFO] Saved MultirangerDeck/multimedia/demo2/wall_distance_demo_plt.png")

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