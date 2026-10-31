"""Base types for specialist SNN agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class AgentOutput:
    """One specialist's contribution to a tick."""

    agent_name: str
    spikes: np.ndarray
    bias_vector: np.ndarray = field(default_factory=lambda: np.zeros(0))
    inhibition_vector: np.ndarray = field(default_factory=lambda: np.zeros(0))
    scalar: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SpecialistAgent(ABC):
    name: str

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def tick(self, context: Dict[str, Any]) -> AgentOutput: ...
