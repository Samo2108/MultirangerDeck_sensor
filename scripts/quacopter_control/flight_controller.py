import math

class QuadcopterController:
    def __init__(self, target_height=0.3, stop_distance=1.0, cruise_pitch=0.1):
        # Mission Goals (Passed from main script)
        self.target_height = target_height
        self.stop_distance = stop_distance
        self.cruise_pitch = cruise_pitch  # Radians of forward tilt

        # Altitude PD Gains
        self.Kp_z = 50.0             
        self.Kd_z = 2.0              

        # Forward Motion & Braking Gains
        self.Kd_vx = 5.0             
        self.pitch_max = 0.08        
        self.max_target_pitch = 0.25 

        # Stabilizer (Angle -> Motor Torque) Gains
        self.Kp_angle = 0.5  
        self.Kd_angle = 0.05 

    def update(self, front_range, down_range, current_pitch, pitch_rate, vx, vz, base_hover_thrust):
        # --- 1. ALTITUDE CONTROLLER ---
        z_err = self.target_height - down_range
        delta_total = (self.Kp_z * z_err) - (self.Kd_z * vz)
        delta_per_motor = delta_total / 4.0

        # --- 2. THE NAVIGATOR (State Machine) ---
        if abs(z_err) > 0.1:
            # Phase 1: Climbing - Brake against any forward drift
            target_pitch = -self.Kd_vx * vx  
        else:
            # Phase 2: Cruising - We are at the correct altitude
            if front_range > self.stop_distance:
                # Wall is far, fly forward at requested pitch!
                target_pitch = self.cruise_pitch
            else:
                # Wall is too close! Slam on the brakes!
                target_pitch = -self.Kd_vx * vx

        # Clamp max tilt for safety
        target_pitch = max(-self.max_target_pitch, min(self.max_target_pitch, target_pitch))

        # --- 3. THE STABILIZER (Angle -> Motor Torque) ---
        pitch_err = target_pitch - current_pitch
        pitch_command = (self.Kp_angle * pitch_err) - (self.Kd_angle * pitch_rate)
        
        # Clamp maximum differential torque
        pitch_command = max(-self.pitch_max, min(self.pitch_max, pitch_command))

        # --- 4. MOTOR MIXER ---
        front_thrust = base_hover_thrust + delta_per_motor - pitch_command
        rear_thrust  = base_hover_thrust + delta_per_motor + pitch_command

        return front_thrust, rear_thrust, target_pitch, pitch_command