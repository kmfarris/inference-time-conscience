"""Core SNN primitives — reservoir, encoding, spike dynamics."""

from brain_regions.core.reservoir import ReservoirSNN
from brain_regions.core.encoding import TokenEncoder

__all__ = ["ReservoirSNN", "TokenEncoder"]
