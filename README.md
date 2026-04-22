# Custom Multiranger Deck Sensor for Isaac Lab

This repository contains the implementation of a custom Time-of-Flight (ToF) Multiranger Deck sensor for NVIDIA Isaac Lab. The sensor simulates a 5-directional distance measurement system (Front, Back, Left, Right, Up/Down), inspired by drone hardware like the Crazyflie Multiranger Deck.

## 1. Installation Requirements

### Hardware Requirements
* **GPU:** NVIDIA RTX GPU (Minimum 8GB VRAM recommended for raycasting simulation).
* **RAM:** 16GB minimum (32GB recommended).

### Software Requirements
* **OS:** Ubuntu 20.04 / 22.04 (or Windows 11 with WSL2).
* **Simulator:** Omniverse Isaac Sim (v2023.1.1 or later).
* **Framework:** NVIDIA Isaac Lab (installed from source).
* **Python:** Python 3.10+.

### Installation
Since this project is packaged with a `setup.py`, you can install it into your Isaac Lab environment as an editable package.
1. Install Isaac Lab following the [official guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
2. Activate your Isaac Lab virtual environment.


### 2. Repository Structure
Project is strictly organized to separate core logic, execution scripts, and documentation:

`
MultirangerDeck/
├── .gitignore                     # Untracked files and cache exclusions
├── README.md                      # Project documentation
├── setup.py                       # Python package installation script
│
├── source/                        # Core Multiranger Deck Sensor Package
│   ├── _init_.py
│   ├── multiranger_deck.py        # Main raycaster sensor class
│   ├── multiranger_deck_cfg.py    # Sensor configurations
│   ├── multiranger_deck_data.py   # Data container for range outputs
│   └── patterns/                  # Raycast pattern generators
│       ├── _init_.py
│       └── multiranger_deck_patterns.py # Math for the 27° 5-cone FoV
│
├── scripts/                       # Executable Isaac Lab Scenarios
│   ├── demo1_wall_validation.py   # Static teleportation accuracy test
│   ├── demo2_wall_validation.py   # Dynamic perfect-hover wall test
│   ├── demo3_pyramid_hover.py     # Forward cruise and altitude test
│   │
│   └── quacopter_control/         # Flight controller logic
│       └── flight_controller.py   # Cascaded PID (Altitude & Pitch)
│
└── multimedia/                    # Output telemetry, plots, and videos
    ├── demo1/
    │   └── wall_distance_demo.png
    ├── demo2/
    │   ├── wall_distance_demo.png
    │   └── wall_distance_demo_plt.png
    └── demo3/
        └── pyramid_hover_telemetry.png
`



### 3. Usage
We have prepared three progressive demonstrations to validate the sensor. To run them, open your terminal, activate the Isaac Lab environment, and execute the scripts from the repository root.

1.  Navigate to the root of isaac lab directory:
   `cd ~/IsaacLab`

2. To run the demo simmulation:
   Demo 1: Basic Wall Validation
   Tests the sensor's basic directional measurements in a static environment.

   `./isaaclab.sh -p /path to your folder/MultirangerDeck/scripts/demo1_wall_validation.py --headless --enable_cameras`

   Demo 2: Offset Wall Algorithm Validation
   Demonstrates the sensor correctly returning the closest hit within a generated 10-ray cone.

   `./isaaclab.sh -p /path to your folder/MultirangerDeck/scripts/demo2_wall_validation.py --headless --enable_cameras`

   Demo 3: Dynamic Pyramid Hover (Terrain Following)
   A dynamic simulation where a drone uses the Z-down sensor reading in a control loop to maintain a stable 30cm altitude over uneven pyramidal terrain.
   
   `./isaaclab.sh -p /path to your folder/MultirangerDeck/scripts/demo3_pyramid_hover.py --headless --enable_cameras`

### 4. Contributions
Alexandru Zaporojanu, Luca Samorì, Tommaso Tieri.

### 5. Credits
Framework: Built using the NVIDIA Isaac Lab framework.

Hardware Inspiration: Logic and configuration inspired by the Bitcraze Crazyflie Multiranger Deck.
