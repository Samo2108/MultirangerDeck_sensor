from isaaclab.sensors.ray_caster.multi_mesh_ray_caster import MultiMeshRayCaster
from collections.abc import Sequence
#from isaaclab.sensors.multiranger_deck.multiranger_deck_cfg import MultirangerDeckCfg
from source.multiranger_deck_data import MultirangerDeckData
import torch
import math

class MultirangerDeck(MultiMeshRayCaster):
    """The Multiranger Deck Sensor Class."""
    cfg: "MultirangerDeckCfg"
    
    def __init__(self, cfg: "MultirangerDeckCfg"):
        super().__init__(cfg)
        self._data = MultirangerDeckData()
        total_rays = self.cfg.pattern_cfg.rays_per_cone
        
        # Replicate the exact same split logic from the pattern generator
        num_in = math.ceil((total_rays - 1) / 3.0)
        num_out = total_rays - 1 - num_in
        
        # Dynamically build the weights list
        weights_list = [5.0] + [0.5] * num_in + [0.1] * num_out
        
        # Create the raw tensor safely before the device is fully initialized
        self._raw_weights = torch.tensor(weights_list)
        
    def _update_buffers_impl(self, env_ids: Sequence[int]):
        super()._update_buffers_impl(env_ids)
        
        if self._data.ranges is None:
            self._data.ranges = torch.zeros(self._view.count, 5, device=self.device)
            # Send the weights to the GPU
            self._ray_weights = self._raw_weights.to(self.device).view(1, 1, -1)
            
        hit_distances = torch.norm(self._data.ray_hits_w[env_ids] - self._ray_starts_w[env_ids], dim=-1)
        grouped_distances = hit_distances.view(-1, 5, self.cfg.pattern_cfg.rays_per_cone)
        
        weighted_dists = grouped_distances * self._ray_weights
        
        # weighted average across the rays in each cone, using the weights
        simulated_mean_dists = torch.sum(weighted_dists, dim=2) / torch.sum(self._ray_weights)
        
        # Clamp
        self._data.ranges[env_ids] = torch.clamp(simulated_mean_dists, max=self.cfg.max_distance)
