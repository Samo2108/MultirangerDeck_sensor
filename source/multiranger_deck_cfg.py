from isaaclab.utils import configclass
from isaaclab.sensors.ray_caster.multi_mesh_ray_caster_cfg import MultiMeshRayCasterCfg
from source.multiranger_deck_data import MultirangerDeckData
from source.patterns.multiranger_deck_patterns import MultirangerPatternCfg
from source.multiranger_deck import MultirangerDeck
@configclass
class MultirangerDeckCfg(MultiMeshRayCasterCfg):
    """Configuration for the Multiranger Deck."""
    class_type: type = MultirangerDeck
    pattern_cfg: MultirangerPatternCfg = MultirangerPatternCfg()