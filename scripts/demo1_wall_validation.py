# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys
import argparse
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)


from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Multiranger Deck - Validation Scenario")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from PIL import Image
import math
import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import CameraCfg
from isaaclab_assets import CRAZYFLIE_CFG

from source.multiranger_deck_cfg import MultirangerDeckCfg
from source.patterns.multiranger_deck_patterns import MultirangerPatternCfg

# Camera Rotations
_x, _y, _z, _w = Rotation.from_euler('xyz', [-180, 0, 0], degrees=True).as_quat()
CUSTOM_CAMERA_ROT = (_w, _x, _y, _z)

RAYCAST_TARGETS = [
    "/World/Ground", 
    "{ENV_REGEX_NS}/Wall.*"
]   #"North", "/World/WallSouth", "/World/WallEast", "/World/WallWest"

@configclass
class ValidationSceneCfg(InteractiveSceneCfg):
    """A mathematically perfect 4x4 meter box for sensor validation."""
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Grid/default_environment.usd")
    )

    # We shift the 0.1m thick walls by 0.05m so the inner faces rest EXACTLY at +/- 2.0m
    wall_north = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WallNorth", # Front (+X)
        spawn=sim_utils.CuboidCfg(size=(0.1, 4.0, 2.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.7))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(2.05, 0.0, 1.0)) 
    )
    wall_south = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WallSouth", # Back (-X)
        spawn=sim_utils.CuboidCfg(size=(0.1, 4.0, 2.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.7))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-2.05, 0.0, 1.0)) 
    )
    wall_east = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WallEast", # Left (+Y)
        spawn=sim_utils.CuboidCfg(size=(4.0, 0.1, 2.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.7))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 2.05, 1.0)) 
    )
    wall_west = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WallWest", # Right (-Y)
        spawn=sim_utils.CuboidCfg(size=(4.0, 0.1, 2.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.7))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -2.05, 1.0))
    )

    robot = CRAZYFLIE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos = (0.0, 0.0, 1.0) 
    
    multiranger = MultirangerDeckCfg(
        prim_path="{ENV_REGEX_NS}/Robot", 
        update_period=1 / 60,
        offset=MultirangerDeckCfg.OffsetCfg(pos=(0, 0, 0)),
        pattern_cfg=MultirangerPatternCfg(
            fov_degrees=15.0,
            #rays_per_cone=1
        ),
        mesh_prim_paths=RAYCAST_TARGETS,
        ray_alignment="yaw",
        max_distance=4.0, 
        debug_vis=True, 
    )
    
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/MainCamera",
        update_period=1 / 30,  height=480, width=640, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955),
        offset=CameraCfg.OffsetCfg(pos=(-0.0, 0.0, 8.0), rot=CUSTOM_CAMERA_ROT),
    )

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot = scene["robot"]
    sim.reset()
    
    # Define our test coordinates [X, Y, Z]
    test_positions = [
        [0.0, 0.0, 1.0],   # Center
        [1.0, 0.0, 1.0],   # 1m Forward
        [0.0, 1.5, 0.5],   # 1.5m Left, Low altitude
        [-1.0, -1.0, 1.5], # 1m Back, 1m Right, High altitude
       # [1.8, 1.8, 0.3]    # Very close to Front-Left corner
    ]

    num_drones = scene.num_envs
    all_drones_history = [[[] for _ in range(num_drones)] for _ in range(num_drones)]
    expected = [] #List of ideal tests

    print("\n" + "="*70)
    print(" MULTIRANGER DECK VALIDATION PROTOCOL")
    print("="*70)

    save_dir = os.path.join(parent_dir, "multimedia", "demo1")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # Data Loggers for plotting
    all_expected = []
    all_measured = []
    all_errors_mm = []

    root_state = robot.data.default_root_state.clone()

    for idx, pos in enumerate(test_positions):
        if not simulation_app.is_running():
            break

        # Teleport the robot to the exact test coordinate
        root_state = robot.data.default_root_state.clone()
        root_state[:, 0:3] = torch.tensor(pos, device=sim.device)
        root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=sim.device) 
        root_state[:, 7:13] = 0.0 
        
        robot.reset()
        scene.reset()

        # step sim for a few frames
        for _ in range(20):
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            
            scene.write_data_to_sim()
            sim.step(render=True)
            
            # Update the scene (which updates the camera and multiranger)
            scene.update(sim.get_physics_dt())

        sim.render()
        scene.update(sim.get_physics_dt())

        # taking picture
        images = scene["camera"].data.output["rgb"]
        if images is not None:
                img_np = images[0].cpu().numpy()
                img = Image.fromarray(img_np.astype('uint8')).convert("RGB")
                
                img.save(f"{save_dir}/test_{idx+1}_view.png")

        # Read the actual sensor data
        ranges = scene["multiranger"].data.ranges

        meas_f = float(ranges[0, 0].item())
        meas_b = float(ranges[0, 1].item())
        meas_l = float(ranges[0, 2].item())
        meas_r = float(ranges[0, 3].item())
        meas_d = float(ranges[0, 4].item())

        # Calculate the mathematically expected distances
        exp_f = 2.0 - pos[0]
        exp_b = pos[0] - (-2.0)
        exp_l = 2.0 - pos[1]
        exp_r = pos[1] - (-2.0)
        exp_d = pos[2] # Ground is exactly at Z=0
        
        directions = [
            (exp_f, meas_f),
            (exp_b, meas_b),
            (exp_l, meas_l),
            (exp_r, meas_r),
            (exp_d, meas_d)
        ]

        for exp, meas in directions:
            error = abs(exp - meas) * 1000 # Convert to millimeters
            if error < 0.01: error = 0.0 

            # Save to logs for plotting
            all_expected.append(exp)
            all_measured.append(meas)
            all_errors_mm.append(error)


    print("\n[INFO] Validation complete. Generating Plots...")

    # GENERATE VALIDATION PLOTS
    if len(all_expected) > 0:
        # Dynamically calculate grid size based on number of tests
        num_tests = len(test_positions)
        cols = 3
        # Add 1 to num_tests to account for the Map, then calculate required rows
        rows = math.ceil((num_tests + 1) / cols) 
        
        # Create the grid of subplots
        fig, axs = plt.subplots(rows, cols, figsize=(20, 6 * rows))
        axs = axs.flatten() # Flatten the 2D array of axes to easily iterate through them

        # --- PLOT 1: Top-Down Spatial Map (Always in the top-left corner) ---
        ax_map = axs[0]
        ax_map.plot([-2, 2, 2, -2, -2], [-2, -2, 2, 2, -2], 'k-', linewidth=3, label='Concrete Walls')
        
        x_coords = [pos[0] for pos in test_positions]
        y_coords = [pos[1] for pos in test_positions]
        z_coords = [pos[2] for pos in test_positions]
        
        ax_map.scatter(x_coords, y_coords, color='red', s=100, edgecolors='black', zorder=3, label='Drone Positions')
        
        for i, (x, y, z) in enumerate(zip(x_coords, y_coords, z_coords)):
            ax_map.annotate(f"T{i+1}\nZ={z}m", (x, y), textcoords="offset points", xytext=(8,8), ha='left', fontsize=10, weight='bold')

        ax_map.set_title('Top-Down Map: Validation Coordinates')
        ax_map.set_xlabel('X Position (meters)')
        ax_map.set_ylabel('Y Position (meters)')
        ax_map.set_aspect('equal', adjustable='box') 
        ax_map.set_xlim(-2.5, 2.5) 
        ax_map.set_ylim(-2.5, 2.5)
        ax_map.grid(True, linestyle='--', alpha=0.4)
        ax_map.legend(loc='upper left', fontsize='small', framealpha=0.8)

        # --- PLOTS 2 to N: Individual Test Charts ---
        directions = ["Front", "Back", "Left", "Right", "Down"]
        x_pos = np.arange(len(directions))
        width = 0.35 # Width of the bars

        for i in range(num_tests):
            ax = axs[i + 1] # Shift index by 1 to skip the Map
            
            # Slice the flat arrays to grab only the 5 data points for THIS specific test
            start_idx = i * 5
            end_idx = start_idx + 5
            
            exp_vals = all_expected[start_idx:end_idx]
            meas_vals = all_measured[start_idx:end_idx]
            err_vals_mm = all_errors_mm[start_idx:end_idx]

            # Primary Axis (Left) - Expected vs Measured in Meters
            rects1 = ax.bar(x_pos - width/2, exp_vals, width, label='Expected (m)', color='royalblue', edgecolor='black')
            rects2 = ax.bar(x_pos + width/2, meas_vals, width, label='Measured (m)', color='darkorange', edgecolor='black')
            
            ax.set_ylabel('Distance (meters)')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(directions)
            
            # Secondary Axis (Right) - Error in Millimeters
            ax2 = ax.twinx()
            ax2.plot(x_pos, err_vals_mm, color='red', marker='o', linestyle='dashed', linewidth=2, markersize=8, label='Error (mm)')
            ax2.set_ylabel('Error (millimeters)', color='red', weight='bold')
            ax2.tick_params(axis='y', labelcolor='red')

            # Formatting
            ax.set_title(f'Test {i+1} Measurements\n(X={test_positions[i][0]}, Y={test_positions[i][1]}, Z={test_positions[i][2]})')
            ax.grid(axis='y', linestyle=':', alpha=0.6)
            
            # Combine the legends from both axes and put them below the chart
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

        # --- CLEANUP ---
        # If the grid has empty boxes at the end (e.g., a 2x3 grid has 6 spots, but only 5 tests + 1 map),
        # this loop finds the empty subplots and deletes them so you don't get blank squares in your image.
        for j in range(num_tests + 1, len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15) # Give extra room at the bottom for the legends
        plt.savefig(f"{save_dir}/wall_distance_demo.png")
        print("[INFO] Plot saved to wall_distance_demo.png!\n")
        
def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0.0, -0.01, 8.0], target=[0.0, 0.0, 0.0]) # Top-down view
    
    scene_cfg = ValidationSceneCfg(num_envs=args_cli.num_envs, env_spacing=6.0)
    scene = InteractiveScene(scene_cfg)
    
    run_simulator(sim, scene)

if __name__ == "__main__":
    main()
    simulation_app.close()