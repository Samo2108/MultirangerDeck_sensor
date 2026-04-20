from dataclasses import dataclass
import torch
from isaaclab.sensors.ray_caster.multi_mesh_ray_caster_data import MultiMeshRayCasterData

@dataclass
class MultirangerDeckData(MultiMeshRayCasterData):
    """Data container holding the 5 final range measurements."""
    ranges: torch.Tensor = None
