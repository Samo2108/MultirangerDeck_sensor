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
        
        # Yaw PD Gains
        self.Kp_yaw = 0.5
        self.Kd_yaw = 0.05
        self.target_yaw = None
        
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
    
    def set_yaw(self, target_yaw):
        self.target_yaw = target_yaw
        
        
    def update(self, down_range, current_pitch, pitch_rate, current_roll, roll_rate, vx, vy, vz, ax, ay, base_hover_thrust, current_yaw, yaw_rate):
        dt = self.sim.get_physics_dt()
        current_time = self.sim.current_time
        
        # ALTITUDE CONTROLLER 
        z_err = self.target_height - down_range
        delta_total = (self.Kp_z * z_err) - (self.Kd_z * vz)
        delta_per_motor = delta_total / 4.0

        # Brake if we aren't at the target height
        # if abs(z_err) > 0.1:
        #     target_vel_x = 0.0 
        # else:
        target_vel_x = self.target_vel

        target_pitch = self.compute_pitch(target_vel_x, vx, ax)
        target_roll  = self.compute_pitch(0.0, -vy, -ay)
        
        # ATTITUDE STABILIZER
        pitch_err = target_pitch - current_pitch
        pitch_command = (self.Kp_angle * pitch_err) - (self.Kd_angle * pitch_rate)
        pitch_command = max(-self.pitch_max, min(self.pitch_max, pitch_command))

        roll_err = target_roll - current_roll
        roll_command = (self.Kp_angle * roll_err) - (self.Kd_angle * roll_rate)
        roll_command = max(-self.pitch_max, min(self.pitch_max, roll_command))

        # YAW CONTROLLER 
        yaw_command = 0.0
        if self.target_yaw is not None:
            # Calculate shortest path
            yaw_err = (self.target_yaw - current_yaw + math.pi) % (2 * math.pi) - math.pi
            yaw_command = (self.Kp_yaw * yaw_err) - (self.Kd_yaw * yaw_rate)
            
            # Safe Clamp: Never use more than 30% of thrust for spinning
            yaw_max = base_hover_thrust * 0.30 
            yaw_command = max(-yaw_max, min(yaw_max, yaw_command))
            

        # m0: Front-Right (CW), m1: Rear-Right (CCW), m2: Rear-Left (CW), m3: Front-Left (CCW)
        m0 = base_hover_thrust + delta_per_motor - pitch_command - roll_command - yaw_command
        m1 = base_hover_thrust + delta_per_motor + pitch_command - roll_command + yaw_command
        m2 = base_hover_thrust + delta_per_motor + pitch_command + roll_command - yaw_command
        m3 = base_hover_thrust + delta_per_motor - pitch_command + roll_command + yaw_command

        if self.debug:
            self.log_time.append(current_time) # Use the internally grabbed time!
            self.log_target_pitch.append(math.degrees(target_pitch))
            self.log_actual_pitch.append(math.degrees(current_pitch))
            self.log_thrust_diff.append(pitch_command * 2.0)
            self.log_x_velocity.append(vx)
            self.log_x_velocity_target.append(target_vel_x)
            self.log_x_acceleration.append(ax)
            self.log_yaw_command.append(yaw_command)
            self.log_actual_yaw.append(current_yaw)
        
        return max(0, m0), max(0, m1), max(0, m2), max(0, m3)
    
    
    def compute_pitch(self, target_vel_x, current_vel_x, current_accel_x):
        # speed controller
        error = target_vel_x - current_vel_x
        pitch_command = (self.Kp_vel * error) -(self.Kd_vel * current_accel_x)
        
        # for safety
        pitch_command = max(-self.max_target_pitch, min(self.max_target_pitch, pitch_command))
        
        return pitch_command
            
    
    def plot_debug(self, save_dir):
        
        if not self.debug or len(self.log_time) == 0:
            return
            
        fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

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
        
        axs[3].plot(self.log_time, self.log_yaw_command, label="Yaw Command", color="magenta")
        axs[3].plot(self.log_time, self.log_actual_yaw, label="Actual Yaw", color="brown")
        axs[3].set_ylabel("Yaw (radians)")
        axs[3].set_xlabel("Simulation Time (seconds)")
        axs[3].set_title("Debug: Yaw Control")
        axs[3].legend()
        axs[3].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(save_dir, "controller_debug_telemetry.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"[INFO] Controller debug plot saved to {plot_path}")