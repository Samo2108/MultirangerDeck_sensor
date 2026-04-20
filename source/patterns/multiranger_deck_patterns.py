from isaaclab.utils import configclass
from isaaclab.sensors.ray_caster.patterns.patterns_cfg import PatternBaseCfg
import torch
import math

def multiranger_pattern(cfg: 'MultirangerPatternCfg', device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates ray directions and origins for a dynamic 5-sensor ToF array."""
    half_fov = math.radians(cfg.fov_degrees) / 2.0
    # Define the directions the sensors are pointing
    base_dirs = torch.tensor([
        [1.0, 0.0, 0.0],  # Front
        [-1.0, 0.0, 0.0], # Back
        [0.0, 1.0, 0.0],  # Left
        [0.0, -1.0, 0.0], # Right
        [0.0, 0.0, -1.0]  # Down
    ], device=device)
 
    # Define the actual positions of the sensors
    d_off = 0.03  
    z_off = -0.00
    base_offsets = torch.tensor([
        [d_off, 0.0, z_off],   
        [-d_off, 0.0, z_off],  
        [0.0, d_off, z_off],   
        [0.0, -d_off, z_off],  
        [0.0, 0.0, z_off]      
    ], device=device)
 
    ray_dirs = []
    ray_starts = []
 
    for i, b_dir in enumerate(base_dirs):
        if abs(b_dir[2]) < 0.99:
            up = torch.tensor([0.0, 0.0, 1.0], device=device)
        else:
            up = torch.tensor([1.0, 0.0, 0.0], device=device)
        right = torch.linalg.cross(b_dir, up)
        right = right / torch.norm(right)
        up_ort = torch.linalg.cross(right, b_dir)
        # Add the 1 Center ray
        ray_dirs.append(b_dir)
        ray_starts.append(base_offsets[i])
        # Calculate the remaining peripheral rays dynamically
        num_edge_rays = cfg.rays_per_cone - 1 
        if num_edge_rays > 0:
            # Divide 360 degrees (2*pi) by the number of outer rays
            angle_step = (2 * math.pi) / num_edge_rays
            for j in range(num_edge_rays):
                angle = j * angle_step
                vec = (b_dir * math.cos(half_fov) + 
                       (right * math.cos(angle) + up_ort * math.sin(angle)) * math.sin(half_fov))
                ray_dirs.append(vec / torch.norm(vec))
                ray_starts.append(base_offsets[i])
 
    return torch.stack(ray_starts), torch.stack(ray_dirs)

# 2. Define the Pattern Config SECOND, passing the func directly
@configclass
class MultirangerPatternCfg(PatternBaseCfg):
    """Configuration for the multiranger 5-cone pattern."""
    func: callable = multiranger_pattern
    rays_per_cone: int = 10  # 1 center ray + 9 peripheral rays
    fov_degrees: float = 27.0