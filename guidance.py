from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set


@dataclass
class GuidanceSignal:
    top_concepts: List[Tuple[str, float]]
    golden_lantern_activations: Dict[str, float]
    golden_lantern_fired: Set[str]
    emotional_valence: float
    topic_clusters: List[str]
    suppressed_concepts: List[str]
    inhibitory_suppression: float = 0.0
    raw_inference: Any = field(repr=False, default=None)
