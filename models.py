# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OpenEnv models for the Trust Game negotiation environment."""

from enum import Enum
from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


class AgentRole(str, Enum):
    """Behavior profile assigned to each negotiating agent."""

    HONEST = "honest"
    VICTIM = "victim"
    SELF_INTERESTED = "self_interested"
    ADVERSARIAL = "adversarial"
    OVERSIGHT = "oversight"


class OversightFlag(BaseModel):
    """Structured feedback emitted by the oversight process."""

    flagged_agent_id: int = Field(..., description="ID of the flagged agent")
    reason: str = Field(..., description="Reason for the flag")
    severity: str = Field(default="info", description="Flag severity")
    details: Dict[str, Any] = Field(
        default_factory=dict, description="Optional structured details"
    )


class NegotiationAction(Action):
    """Action taken by an agent during one negotiation step."""

    agent_id: int = Field(..., description="ID of the acting agent")
    claim_amount: float = Field(..., ge=0, le=100, description="Resource amount claimed")
    verify_targets: List[int] = Field(
        default_factory=list, description="Agent IDs to verify"
    )
    accept_proposal: bool = Field(False, description="Accept current resource split")
    message: Optional[str] = Field(None, description="Optional justification message")


class NegotiationObservation(Observation):
    """Observation emitted after each negotiation step."""

    round_number: int = Field(..., ge=0)
    your_agent_id: int
    your_role: AgentRole
    your_true_need: float = Field(..., ge=0, le=100)

    all_claims: Dict[int, float] = Field(
        default_factory=dict, description="Latest claims from all agents"
    )
    trust_scores: Dict[int, float] = Field(
        default_factory=dict, description="Your trust toward each agent"
    )
    claim_history: Dict[int, List[float]] = Field(
        default_factory=dict, description="Recent claim history per agent"
    )

    oversight_flags: List[OversightFlag] = Field(
        default_factory=list, description="Oversight feedback from this step"
    )
    allocation: Optional[Dict[int, float]] = Field(
        default=None, description="Final allocation when a round is complete"
    )
    negotiation_complete: bool = Field(
        default=False, description="Whether the current negotiation has completed"
    )
    system_metrics: Dict[str, float] = Field(
        default_factory=dict, description="System-level trust/deception/fairness metrics"
    )
    prompt: str = Field(..., description="Prompt for the next action")


class EpisodeState(State):
    """State returned by the OpenEnv `state()` endpoint."""

    total_rounds: int = Field(..., ge=1)
    curriculum_stage: int = Field(default=0, ge=0)
    agents_config: Dict[int, AgentRole] = Field(default_factory=dict)
    system_metrics: Dict[str, float] = Field(
        default_factory=dict, description="Episode-level aggregate metrics"
    )


class TrustGameAction(NegotiationAction):
    """Backward-compatible alias used by existing server/client imports."""


class TrustGameObservation(NegotiationObservation):
    """Backward-compatible alias used by existing server/client imports."""
