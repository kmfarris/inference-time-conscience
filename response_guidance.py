from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ResponseGuidance:
    conversational_mode: str = "general"
    target_sentence_count: int = 3
    target_sentence_length: str = "medium"
    energy_level: float = 0.5
    personal_investment: float = 0.0
    concept_priority: List[Tuple[str, float]] = field(default_factory=list)
    confidence: float = 0.5
    intent_summary: str = ""
