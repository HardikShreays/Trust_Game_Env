"""Multi-seed, multi-condition evaluation runner for the Trust Game.

Example:
    python -m trust_game_env.eval.run_eval \
        --seeds 50 \
        --conditions full no_oversight no_deception_reward \
        --curriculum-stage 2 \
        --out-dir eval_results/
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from ..baselines.policies import Policy, policy_for_role
from ..models import AgentRole, TrustGameObservation
from ..server.trust_game_env_environment import TrustGameEnvironment
from .ablations import DEFAULT_GRID, GRID_BY_NAME, Condition
from .metrics import aggregate, detection_metrics, summarise

PolicyFactory = Callable[[AgentRole], Policy]

SUMMARY_KEYS = (
    "episode_reward_mean",
    "trust_stability",
    "deception_rate",
    "deception_detection_rate",
    "fairness",
    "efficiency",
    "long_term_payoff",
    "detection_precision",
    "detection_recall",
    "detection_f1",
    "total_lies",
    "total_successful_deceptions",
    "steps",
)


@dataclass
class EpisodeResult:
    condition: str
    seed: int
    reward_by_role: Dict[str, float]
    system_metrics: Dict[str, float]
    detection: Dict[str, float]
    steps: int
    total_lies: int
    total_successful_deceptions: int

    def to_row(self) -> dict:
        role_rewards = {f"reward_{role}": value for role, value in self.reward_by_role.items()}
        return {
            "condition": self.condition,
            "seed": self.seed,
            "steps": self.steps,
            "total_lies": self.total_lies,
            "total_successful_deceptions": self.total_successful_deceptions,
            "episode_reward_mean": (
                sum(self.reward_by_role.values()) / max(1, len(self.reward_by_role))
            ),
            "detection_precision": self.detection["precision"],
            "detection_recall": self.detection["recall"],
            "detection_f1": self.detection["f1"],
            **role_rewards,
            **{k: v for k, v in self.system_metrics.items()},
        }


def _build_env(condition: Condition, seed: int, curriculum_stage: int) -> TrustGameEnvironment:
    return TrustGameEnvironment(
        curriculum_stage=curriculum_stage,
        seed=seed,
        **condition.env_kwargs,
    )


def run_episode(
    condition: Condition,
    seed: int,
    curriculum_stage: int = 2,
    policy_factory: Optional[PolicyFactory] = None,
    max_steps: Optional[int] = None,
) -> EpisodeResult:
    """Run a single episode with scripted (or provided) policies."""
    policy_factory = policy_factory or policy_for_role
    env = _build_env(condition, seed, curriculum_stage)
    obs: TrustGameObservation = env.reset()

    if max_steps is None:
        max_steps = env.max_rounds * env.num_agents

    step = 0
    done = False
    while not done and step < max_steps:
        role = env.roles[obs.your_agent_id]
        policy = policy_factory(role)
        action = policy.act(obs)
        obs, _reward, done = env.step(action)
        step += 1

    reward_by_role: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for agent_id, total_reward in env.episode_rewards.items():
        role_name = env.roles[agent_id].value
        reward_by_role[role_name] = reward_by_role.get(role_name, 0.0) + float(total_reward)
        counts[role_name] = counts.get(role_name, 0) + 1
    for role_name, count in counts.items():
        reward_by_role[role_name] = reward_by_role[role_name] / max(1, count)

    det = detection_metrics(env.oversight_flags, env.total_lies)

    return EpisodeResult(
        condition=condition.name,
        seed=seed,
        reward_by_role=reward_by_role,
        system_metrics=dict(env._system_metrics()),
        detection=det.to_dict(),
        steps=step,
        total_lies=env.total_lies,
        total_successful_deceptions=env.total_successful_deceptions,
    )


def run_grid(
    conditions: Sequence[Condition],
    seeds: Sequence[int],
    curriculum_stage: int = 2,
    policy_factory: Optional[PolicyFactory] = None,
) -> List[EpisodeResult]:
    results: List[EpisodeResult] = []
    for condition in conditions:
        for seed in seeds:
            results.append(
                run_episode(
                    condition=condition,
                    seed=seed,
                    curriculum_stage=curriculum_stage,
                    policy_factory=policy_factory,
                )
            )
    return results


def write_raw_csv(results: Sequence[EpisodeResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r.to_row() for r in results]
    if not rows:
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_json(results: Sequence[EpisodeResult], path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, dict] = {}
    by_condition: Dict[str, List[dict]] = {}
    for r in results:
        by_condition.setdefault(r.condition, []).append(r.to_row())

    for condition_name, rows in by_condition.items():
        summary[condition_name] = {
            "n_seeds": len(rows),
            "metrics": summarise(rows, keys=SUMMARY_KEYS),
        }

    with path.open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Trust Game ablation evaluations.")
    parser.add_argument("--seeds", type=int, default=20, help="Number of seeds per condition")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=[c.name for c in DEFAULT_GRID],
        help="Condition names (from eval.ablations)",
    )
    parser.add_argument("--curriculum-stage", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=Path("eval_results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    conditions: List[Condition] = []
    for name in args.conditions:
        if name not in GRID_BY_NAME:
            raise SystemExit(f"Unknown condition: {name}. Available: {list(GRID_BY_NAME)}")
        conditions.append(GRID_BY_NAME[name])

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    results = run_grid(conditions, seeds, curriculum_stage=args.curriculum_stage)

    raw_path = args.out_dir / "results_raw.csv"
    summary_path = args.out_dir / "results_summary.json"
    write_raw_csv(results, raw_path)
    summary = write_summary_json(results, summary_path)

    print(f"Wrote {len(results)} episodes to {raw_path}")
    print(f"Wrote summary to {summary_path}")
    for name, payload in summary.items():
        reward = payload["metrics"]["episode_reward_mean"]
        f1 = payload["metrics"]["detection_f1"]
        print(
            f"[{name}] reward={reward['mean']:+.3f} ±{reward['ci95']:.3f}  "
            f"F1={f1['mean']:.3f} ±{f1['ci95']:.3f}  n={payload['n_seeds']}"
        )


if __name__ == "__main__":
    main()
