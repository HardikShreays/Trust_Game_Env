"""Scripted baseline policies for every role in the Trust Game.

These policies produce `TrustGameAction` objects from a `TrustGameObservation`.
They are intentionally simple so they can be used as a reference point for
comparing against learned policies (e.g. RL-trained LLMs).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from ..models import AgentRole, TrustGameAction, TrustGameObservation


@dataclass
class Policy:
    """Base scripted policy. Subclasses override `act`."""

    name: str = "base"

    def act(self, obs: TrustGameObservation) -> TrustGameAction:  # pragma: no cover
        raise NotImplementedError

    def _clip(self, value: float) -> float:
        return float(np.clip(value, 0.0, 100.0))

    def _lowest_trust_targets(self, obs: TrustGameObservation, k: int) -> List[int]:
        if not obs.trust_scores:
            return []
        others = [
            (agent_id, score)
            for agent_id, score in obs.trust_scores.items()
            if agent_id != obs.your_agent_id
        ]
        others.sort(key=lambda pair: pair[1])
        return [agent_id for agent_id, _ in others[:k]]


class HonestPolicy(Policy):
    """Claims true need. Verifies the two least-trusted agents. Accepts fair splits."""

    name: str = "honest"

    def act(self, obs: TrustGameObservation) -> TrustGameAction:
        claim = self._clip(obs.your_true_need)
        verify_targets = self._lowest_trust_targets(obs, k=2)
        accept = obs.system_metrics.get("fairness", 0.0) >= 0.7
        return TrustGameAction(
            agent_id=obs.your_agent_id,
            claim_amount=claim,
            verify_targets=verify_targets,
            accept_proposal=accept,
            communication_act="reassure",
            message="claiming true need",
        )


class SelfInterestedPolicy(Policy):
    """Moderately inflates claim. Avoids large deviations to dodge detection."""

    name: str = "self_interested"
    inflation: float = 10.0

    def act(self, obs: TrustGameObservation) -> TrustGameAction:
        claim = self._clip(obs.your_true_need + self.inflation)
        return TrustGameAction(
            agent_id=obs.your_agent_id,
            claim_amount=claim,
            verify_targets=[],
            accept_proposal=True,
            communication_act="false_promise",
            message="prioritising my share",
        )


class AdversarialPolicy(Policy):
    """Strong inflation with low variance (stealth). Accepts when favourable."""

    name: str = "adversarial"
    inflation: float = 25.0

    def act(self, obs: TrustGameObservation) -> TrustGameAction:
        # Keep variance low: always inflate by a fixed amount.
        claim = self._clip(obs.your_true_need + self.inflation)
        accept = True
        return TrustGameAction(
            agent_id=obs.your_agent_id,
            claim_amount=claim,
            verify_targets=[],
            accept_proposal=accept,
            communication_act="guilt",
            message="we all deserve our share",
        )


class VictimPolicy(Policy):
    """Claims true need. Only verifies once trust drops significantly."""

    name: str = "victim"
    verify_threshold: float = 0.4

    def act(self, obs: TrustGameObservation) -> TrustGameAction:
        claim = self._clip(obs.your_true_need)
        verify_targets = [
            agent_id
            for agent_id, trust in obs.trust_scores.items()
            if agent_id != obs.your_agent_id and trust < self.verify_threshold
        ]
        return TrustGameAction(
            agent_id=obs.your_agent_id,
            claim_amount=claim,
            verify_targets=verify_targets,
            accept_proposal=True,
            communication_act="reassure",
            message="trusting you",
        )


class OversightPolicy(Policy):
    """Claims true need, verifies the top suspected liars, withholds approval."""

    name: str = "oversight"
    verify_k: int = 2

    def act(self, obs: TrustGameObservation) -> TrustGameAction:
        claim = self._clip(obs.your_true_need)
        verify_targets = self._lowest_trust_targets(obs, k=self.verify_k)
        # Oversight accepts only if deception rate is low this episode.
        accept = obs.system_metrics.get("deception_rate", 0.0) < 0.1
        return TrustGameAction(
            agent_id=obs.your_agent_id,
            claim_amount=claim,
            verify_targets=verify_targets,
            accept_proposal=accept,
            communication_act="partial_confession",
            message="monitoring for deception",
        )


def policy_for_role(role: AgentRole) -> Policy:
    """Return the default scripted policy for a given role."""
    mapping = {
        AgentRole.HONEST: HonestPolicy(),
        AgentRole.SELF_INTERESTED: SelfInterestedPolicy(),
        AgentRole.ADVERSARIAL: AdversarialPolicy(),
        AgentRole.VICTIM: VictimPolicy(),
        AgentRole.OVERSIGHT: OversightPolicy(),
    }
    return mapping[role]
