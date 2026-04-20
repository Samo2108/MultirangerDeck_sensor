# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys
import argparse
import time

# --- THE IMPORT FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
# ----------------------

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

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import CameraCfg
from isaaclab_assets import CRAZYFLIE_CFG

from source.multiranger_deck_cfg import MultirangerDeckCfg

# Camera Rotations
_w, _x, _y, _z = Rotation.from_euler('xyz', [-180, 0, 0], degrees=True).as_quat()
CUSTOM_CAMERA_ROT = (_w, _x, _y, _z)

RAYCAST_TARGETS = [
    "/World/Ground", 
    "/World/WallNorth", "/World/WallSouth", 
    "/World/WallEast", "/World/WallWest"
]

@configclass
class ValidationSceneCfg(InteractiveSceneCfg):
    """A mathematically perfect 4x4 meter box for sensor validation."""
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Grid/default_environment.usd")
    )

    # We shift the 0.1m thick walls by 0.05m so the inner faces rest EXACTLY at +/- 2.0m
    wall_north = AssetBaseCfg(
        prim_path="/World/WallNorth", # Front (+X)
        spawn=sim_utils.CuboidCfg(size=(0.1, 4.0, 2.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.7))), 
        init_state=AssetBaseCfg.InitialStateCfg(pos=(2.05, 0.0, 1.0)) 
    )
    wall_south = AssetBaseCfg(
        prim_path="/World/WallSouth", # Back (-X)
        spawn=sim_utils.CuboidCfg(size=(0.1, 4.0, 2.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.7))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-2.05, 0.0, 1.0)) 
    )
    wall_east = AssetBaseCfg(
        prim_path="/World/WallEast", # Left (+Y)
        spawn=sim_utils.CuboidCfg(size=(4.0, 0.1, 2.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.7))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 2.05, 1.0)) 
    )
    wall_west = AssetBaseCfg(
        prim_path="/World/WallWest", # Right (-Y)
        spawn=sim_utils.CuboidCfg(size=(4.0, 0.1, 2.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.7))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -2.05, 1.0))
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
    robot = scene["robot"]
    sim.reset()
    
    # Define our test coordinates [X, Y, Z]
    test_positions = [
        [0.0, 0.0, 1.0],   # Center
        [1.0, 0.0, 1.0],   # 1m Forward
        [0.0, 1.5, 0.5],   # 1.5m Left, Low altitude
        [-1.0, -1.0, 1.5], # 1m Back, 1m Right, High altitude
        [1.8, 1.8, 0.2]    # Very close to Front-Left corner
    ]

    print("\n" + "="*70)
    print(" MULTIRANGER DECK VALIDATION PROTOCOL")
    print("="*70)

    if not os.path.exists("MultirangerDeck/multimedia/demo1/"):
        os.makedirs("MultirangerDeck/multimedia/demo1/")
        
    # Data Loggers for plotting
    all_expected = []
    all_measured = []
    all_errors_mm = []
    test_labels = []

    try:
        for idx, pos in enumerate(test_positions):
            if not simulation_app.is_running():
                break

            # 1. Teleport the robot to the exact test coordinate
            root_state = robot.data.default_root_state.clone()
            root_state[:, 0:3] = torch.tensor(pos, device=sim.device)
            # Reset orientation to perfectly flat
            root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=sim.device) 
            # Zero out all velocities
            root_state[:, 7:] = 0.0 
            
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            robot.reset()
            scene.reset()

            # 2. Step the physics engine briefly to let the raycaster update its hits
            for _ in range(10):
                scene.write_data_to_sim()
                sim.step(render=True)
                scene.update(sim.get_physics_dt())

            # 3. Read the actual sensor data
            ranges = scene["multiranger"].data.ranges[0]
            meas_f = float(ranges[0].item())
            meas_b = float(ranges[1].item())
            meas_l = float(ranges[2].item())
            meas_r = float(ranges[3].item())
            meas_d = float(ranges[4].item())

            # 4. Calculate the mathematically expected distances
            exp_f = 2.0 - pos[0]
            exp_b = pos[0] - (-2.0)
            exp_l = 2.0 - pos[1]
            exp_r = pos[1] - (-2.0)
            exp_d = pos[2] # Ground is exactly at Z=0

            # 5. Print the Validation Report
            print(f"\n TEST #{idx+1} | Drone Coordinates: X={pos[0]}, Y={pos[1]}, Z={pos[2]}")
            print("-" * 70)
            print(f"{'Direction':<10} | {'Expected (m)':<15} | {'Measured (m)':<15} | {'Error (mm)':<10}")
            print("-" * 70)
            
            directions = [
                ("Front", exp_f, meas_f),
                ("Back",  exp_b, meas_b),
                ("Left",  exp_l, meas_l),
                ("Right", exp_r, meas_r),
                ("Down",  exp_d, meas_d)
            ]

            for name, exp, meas in directions:
                error = abs(exp - meas) * 1000 # Convert to millimeters
                if error < 0.01: error = 0.0 
                
                print(f"{name:<10} | {exp:<15.3f} | {meas:<15.3f} | {error:<10.2f}")
                
                # Save to logs for plotting
                all_expected.append(exp)
                all_measured.append(meas)
                all_errors_mm.append(error)
                test_labels.append(f"T{idx+1}-{name[0]}") # e.g., "T1-F" for Test 1 Front

            # Pause briefly so you can see the robot move in the UI
            time.sleep(1)

    except (KeyboardInterrupt, SystemExit):
        pass

    print("\n[INFO] Validation complete. Generating Plots...")

    # ========================================================
    # 6. GENERATE VALIDATION PLOTS
    # ========================================================
    if len(all_expected) > 0:
        # Changed to 1x3 layout for the new map!
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        # --- PLOT 1: Parity Plot (Expected vs Measured) ---
        axs[0].scatter(all_expected, all_measured, color='blue', alpha=0.7, s=60, edgecolors='black', zorder=3)
        max_val = max(max(all_expected), max(all_measured)) + 0.5
        axs[0].plot([0, max_val], [0, max_val], 'r--', label='Ideal (Zero Error)', zorder=2)
        axs[0].set_title('Sensor Accuracy: Expected vs. Measured')
        axs[0].set_xlabel('Expected True Distance (meters)')
        axs[0].set_ylabel('Measured Sensor Distance (meters)')
        axs[0].legend()
        axs[0].grid(True, linestyle=':', alpha=0.6)
        axs[0].set_xlim(0, max_val)
        axs[0].set_ylim(0, max_val)

        # --- PLOT 2: Absolute Error Distribution ---
        bars = axs[1].bar(test_labels, all_errors_mm, color='darkred', zorder=3)
        axs[1].set_title('Measurement Error per Raycast')
        axs[1].set_ylabel('Absolute Error (millimeters)')
        axs[1].tick_params(axis='x', rotation=45)
        axs[1].grid(axis='y', linestyle=':', alpha=0.6)
        
        for bar in bars:
            yval = bar.get_height()
            if yval > 0.1:
                axs[1].text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.1f}', ha='center', va='bottom', fontsize=8)

        # --- PLOT 3: Top-Down Spatial Map ---
        # Draw the 4x4 meter walls (-2 to +2 on both axes)
        axs[2].plot([-2, 2, 2, -2, -2], [-2, -2, 2, 2, -2], 'k-', linewidth=3, label='Concrete Walls')
        
        # Extract the X and Y coordinates from your test_positions array
        x_coords = [pos[0] for pos in test_positions]
        y_coords = [pos[1] for pos in test_positions]
        
        # Plot the drone positions
        axs[2].scatter(x_coords, y_coords, color='red', s=100, edgecolors='black', zorder=3, label='Drone Positions')
        
        # Label each dot with its Test Number (T1, T2, etc.)
        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            axs[2].annotate(f"T{i+1}", (x, y), textcoords="offset points", xytext=(8,8), ha='left', fontsize=10, weight='bold')

        axs[2].set_title('Top-Down Map: Validation Coordinates')
        axs[2].set_xlabel('X Position (meters)')
        axs[2].set_ylabel('Y Position (meters)')
        
        # Force the graph to be a perfect square so it doesn't look stretched!
        axs[2].set_aspect('equal', adjustable='box') 
        axs[2].set_xlim(-2.5, 2.5) # Add a 0.5m visual padding outside the walls
        axs[2].set_ylim(-2.5, 2.5)
        axs[2].grid(True, linestyle='--', alpha=0.4)
        axs[2].legend(loc='upper right')

        plt.tight_layout()
        plt.savefig("MultirangerDeck/multimedia/demo1/wall_distance_demo.png")
        print("[INFO] Plot saved to wall_distance_demo.png!\n")

def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0.0, -0.01, 8.0], target=[0.0, 0.0, 0.0]) # Top-down view
    
    scene_cfg = ValidationSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    run_simulator(sim, scene)

if __name__ == "__main__":
    main()
    simulation_app.close()