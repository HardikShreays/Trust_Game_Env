# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Trust Game negotiation environment client."""

from typing import Dict, List

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import AgentRole, EpisodeState, OversightFlag, TrustGameAction, TrustGameObservation


class TrustGameEnv(
    EnvClient[TrustGameAction, TrustGameObservation, EpisodeState]
):
    """
    Client for the Trust Game negotiation environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with TrustGameEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.prompt)
        ...
        ...     result = client.step(
        ...         TrustGameAction(agent_id=0, claim_amount=42.0, accept_proposal=False)
        ...     )
        ...     print(result.observation.all_claims)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = TrustGameEnv.from_docker_image("trust_game_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(
        ...         TrustGameAction(agent_id=0, claim_amount=50.0, accept_proposal=False)
        ...     )
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: TrustGameAction) -> Dict:
        """
        Convert TrustGameAction to JSON payload for step message.

        Args:
            action: TrustGameAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "agent_id": action.agent_id,
            "claim_amount": action.claim_amount,
            "verify_targets": action.verify_targets,
            "accept_proposal": action.accept_proposal,
            "message": action.message,
        }

    def _parse_result(self, payload: Dict) -> StepResult[TrustGameObservation]:
        """
        Parse server response into StepResult[TrustGameObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with TrustGameObservation
        """
        obs_data: Dict = payload.get("observation", {})
        oversight_flags_payload: List[Dict] = obs_data.get("oversight_flags", [])
        oversight_flags = [
            OversightFlag(
                flagged_agent_id=flag.get("flagged_agent_id", -1),
                reason=flag.get("reason", ""),
                severity=flag.get("severity", "info"),
                details=flag.get("details", {}),
            )
            for flag in oversight_flags_payload
        ]

        observation = TrustGameObservation(
            round_number=obs_data.get("round_number", 0),
            your_agent_id=obs_data.get("your_agent_id", 0),
            your_role=AgentRole(obs_data.get("your_role", AgentRole.HONEST.value)),
            your_true_need=obs_data.get("your_true_need", 0.0),
            all_claims=obs_data.get("all_claims", {}),
            trust_scores=obs_data.get("trust_scores", {}),
            claim_history=obs_data.get("claim_history", {}),
            oversight_flags=oversight_flags,
            allocation=obs_data.get("allocation"),
            negotiation_complete=obs_data.get("negotiation_complete", False),
            system_metrics=obs_data.get("system_metrics", {}),
            prompt=obs_data.get("prompt", ""),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> EpisodeState:
        """
        Parse server response into EpisodeState object.

        Args:
            payload: JSON response from state request

        Returns:
            EpisodeState with episode metadata
        """
        agents_config_raw = payload.get("agents_config", {})
        agents_config = {
            int(agent_id): AgentRole(role)
            for agent_id, role in agents_config_raw.items()
        }

        return EpisodeState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            total_rounds=payload.get("total_rounds", 1),
            curriculum_stage=payload.get("curriculum_stage", 0),
            agents_config=agents_config,
            system_metrics=payload.get("system_metrics", {}),
        )
