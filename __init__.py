# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Trust Game: a multi-agent negotiation environment for studying
trust, deception, and oversight in AI agents (OpenEnv compatible)."""

from .client import TrustGameEnv
from .models import (
    AgentRole,
    EpisodeState,
    NegotiationAction,
    NegotiationObservation,
    OversightFlag,
    TrustGameAction,
    TrustGameObservation,
)

__all__ = [
    "TrustGameEnv",
    "TrustGameAction",
    "TrustGameObservation",
    "NegotiationAction",
    "NegotiationObservation",
    "EpisodeState",
    "OversightFlag",
    "AgentRole",
]
