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

### Setup Instructions
Since this project is packaged with a `setup.py`, you can install it into your Isaac Lab environment as an editable package.
1. Activate your Isaac Lab virtual environment.
2. Navigate to the root of this repository.
3. Run the installation command:
   ```bash
   pip install -e .

### 2. Repository Structure
Our project is strictly organized to separate core logic, execution scripts, and documentation:

source/: Contains the core sensor implementation.

- multiranger_deck.py, multiranger_deck_cfg.py, multiranger_deck_data.py: The main sensor classes     and data buffers extending Isaac Lab's RayCaster.

- patterns/multiranger_deck_patterns.py: The mathematical logic generating the 5-cone ray             distribution.

scripts/: Contains the executable demonstrations and controllers.

- quacopter_control/: Contains the flight_controller.py used for dynamic movement overrides.

- demo1_wall_validation.py: Static validation against simple walls.

- demo2_wall_validation.py: Validation of the ToF minimum-distance algorithm using offset walls.

- demo3_pyramid_hover.py: Dynamic terrain-following validation over pyramids.

multimedia/: Contains telemetry plots and screenshots of the simulations in action.

presentation/: Contains the final project slides (Original PPTX and PDF formats).

### 3. Demo Instructions
We have prepared three progressive demonstrations to validate the sensor. To run them, open your terminal, activate the Isaac Lab environment, and execute the scripts from the repository root.

Demo 1: Basic Wall Validation
Tests the sensor's basic directional measurements in a static environment.

Bash
python scripts/demo1_wall_validation.py

Demo 2: Offset Wall Algorithm Validation
Demonstrates the sensor correctly returning the closest hit within a generated 10-ray cone.

Bash python scripts/demo2_wall_validation.py

Demo 3: Dynamic Pyramid Hover (Terrain Following)
A dynamic simulation where a drone uses the Z-down sensor reading in a control loop to maintain a stable 30cm altitude over uneven pyramidal terrain.

Bash python scripts/demo3_pyramid_hover.py

### 4. Contributions
Alexandru Zaporojanu, Luca Samorì, Tommaso Tieri.

### 5. Credits
Framework: Built using the NVIDIA Isaac Lab framework.

Hardware Inspiration: Logic and configuration inspired by the Bitcraze Crazyflie Multiranger Deck.
