import math
import isaaclab.sim as sim_utils
import matplotlib.pyplot as plt
import os

class QuadcopterController:
    def __init__(self, sim: sim_utils.SimulationContext, target_height=0.3, target_vel=0.1, debug=False):
        self.target_height = target_height
        self.target_vel = target_vel
        self.sim = sim
        self.debug = debug
        
        # Altitude PD Gains
        self.Kp_z = 25.0   #10          
        self.Kd_z = 1.5    # 4         

        # Velocity PD Gains
        self.Kp_vel = 1.5        
        self.Kd_vel = 0.001         
        self.max_target_pitch = 0.25 

        # Stabilizer PD Gains
        self.Kp_angle = 0.5
        self.Kd_angle = 0.015
        self.pitch_max = 0.08 
        
        
        if self.debug:
            self.log_time = []
            self.log_target_pitch = []
            self.log_actual_pitch = []
            self.log_thrust_diff = []
            self.log_x_velocity = []
            self.log_x_acceleration = []
            self.log_x_velocity_target = []
            self.log_yaw_command = []
            self.log_actual_yaw = []

    def set_cruise_velocity(self, cruise_vel):
        self.target_vel = cruise_vel
        
    def update(self, down_range, current_pitch, pitch_rate, vx, vz, ax, base_hover_thrust):
        dt = self.sim.get_physics_dt()
        current_time = self.sim.current_time
        
        # ALTITUDE CONTROLLER 
        z_err = self.target_height - down_range
        delta_total = (self.Kp_z * z_err) - (self.Kd_z * vz)
        delta_per_motor = delta_total / 4.0

        # move only when height is at target
        if abs(z_err) > 0.1:
            # Target velocity is 0 to brake against drift
            target_vel_x = 0.0 
        else:
            target_vel_x = self.target_vel

        # VELOCITY CONTROLLER (Velocity -> Target Pitch) 
        target_pitch = self.compute_pitch(target_vel_x, vx, ax, dt) 
        
           
        # add the height control effect to the velocity controller
        pitch_err = target_pitch - current_pitch
        pitch_command = (self.Kp_angle * pitch_err) - (self.Kd_angle * pitch_rate)
        
        # Clamp pitch
        pitch_command = max(-self.pitch_max, min(self.pitch_max, pitch_command))

        # MOTORS
        front_thrust = base_hover_thrust + delta_per_motor - pitch_command
        rear_thrust  = base_hover_thrust + delta_per_motor + pitch_command

        if self.debug:
            self.log_time.append(current_time) # Use the internally grabbed time!
            self.log_target_pitch.append(math.degrees(target_pitch))
            self.log_actual_pitch.append(math.degrees(current_pitch))
            self.log_thrust_diff.append(pitch_command * 2.0)
            self.log_x_velocity.append(vx)
            self.log_x_velocity_target.append(target_vel_x)
            self.log_x_acceleration.append(ax)
        return front_thrust, rear_thrust
    
    
    def compute_pitch(self, target_vel_x, current_vel_x, current_accel_x):
        # speed controller
        error = target_vel_x - current_vel_x
        pitch_command = (self.Kp_vel * error) + (self.Kd_vel * current_accel_x)
        
        # for safety
        pitch_command = max(-self.max_target_pitch, min(self.max_target_pitch, pitch_command))
        
        return pitch_command
            
    
    def plot_debug(self, save_dir):
        
        if not self.debug or len(self.log_time) == 0:
            return
            
        fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

        axs[0].plot(self.log_time, self.log_target_pitch, label="Target Pitch (deg)", linestyle="--", color="orange")
        axs[0].plot(self.log_time, self.log_actual_pitch, label="Actual Pitch (deg)", color="green")
        axs[0].set_ylabel("Angle (degrees)")
        axs[0].set_title("Debug: Pitch Controller Tracking")
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(self.log_time, self.log_thrust_diff, label="Differential Thrust Cmd (N)", color="purple")
        axs[1].set_ylabel("Force (Newtons)")
        axs[1].set_xlabel("Simulation Time (seconds)")
        axs[1].set_title("Debug: Motor Effort")
        axs[1].legend()
        axs[1].grid(True)

        # apply acceleration on the left y-axis and velocity on the right y-axis
        ax2 = axs[2].twinx()
        axs[2].plot(self.log_time, self.log_x_velocity_target, label="Target X Velocity (m/s)", linestyle="--", color="red")
        axs[2].plot(self.log_time, self.log_x_velocity, label="Actual X Velocity (m/s)", color="blue")
        ax2.plot(self.log_time, self.log_x_acceleration, label="X Acceleration (m/s²)", color="cyan")

        axs[2].set_ylabel("Velocity (m/s)")
        ax2.set_ylabel("Acceleration (m/s²)")
        axs[2].set_xlabel("Simulation Time (seconds)")
        axs[2].set_title("Debug: X Velocity")
        axs[2].legend()
        axs[2].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(save_dir, "controller_debug_telemetry.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"[INFO] Controller debug plot saved to {plot_path}")