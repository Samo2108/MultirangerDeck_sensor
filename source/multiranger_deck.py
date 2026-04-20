from isaaclab.sensors.ray_caster.multi_mesh_ray_caster import MultiMeshRayCaster
from collections.abc import Sequence
#from isaaclab.sensors.multiranger_deck.multiranger_deck_cfg import MultirangerDeckCfg
from source.multiranger_deck_data import MultirangerDeckData
import torch

class MultirangerDeck(MultiMeshRayCaster):
    """The Multiranger Deck Sensor Class."""
    cfg: "MultirangerDeckCfg"
    
    def __init__(self, cfg: "MultirangerDeckCfg"):
        super().__init__(cfg)
        self._data = MultirangerDeckData()
        
    def _update_buffers_impl(self, env_ids: Sequence[int]):
        super()._update_buffers_impl(env_ids)
        
        if self._data.ranges is None:
            self._data.ranges = torch.zeros(self._view.count, 5, device=self.device)
            
        hit_distances = torch.norm(self._data.ray_hits_w[env_ids] - self._ray_starts_w[env_ids], dim=-1)
        grouped_distances = hit_distances.view(-1, 5, self.cfg.pattern_cfg.rays_per_cone)
        self._data.ranges[env_ids] = torch.min(grouped_distances, dim=2)[0]
