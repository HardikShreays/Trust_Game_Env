"""Predefined ablation grid for experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class Condition:
    """One experimental condition (env flag overrides)."""

    name: str
    env_kwargs: Dict[str, bool] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"name": self.name, **self.env_kwargs}


DEFAULT_GRID: List[Condition] = [
    Condition(name="full"),
    Condition(name="no_oversight", env_kwargs={"enable_oversight": False}),
    Condition(
        name="no_deception_reward",
        env_kwargs={"enable_deception_sparse_reward": False},
    ),
    Condition(name="no_trust_updates", env_kwargs={"enable_trust_updates": False}),
    Condition(name="no_belief_updates", env_kwargs={"enable_belief_updates": False}),
]


GRID_BY_NAME: Dict[str, Condition] = {c.name: c for c in DEFAULT_GRID}
