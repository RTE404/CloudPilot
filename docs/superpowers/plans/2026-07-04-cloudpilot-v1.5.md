# CloudPilot v1.5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert CloudPilot from a homogeneous, single-objective scheduler into a heterogeneous, multi-objective one (capacity tiers, priority/SLA jobs, multi-term reward), and prove a retrained PPO agent beats a priority-aware Least Loaded baseline on SLA violation rate without starving Bronze jobs.

**Architecture:** Full dict-based data model rename (`Job.requirements`/`Server.capacity`), a new global waiting queue replacing v1's dead per-server queues, a new `environment/priority.py` module owning tier/SLA/reward logic, nested Pydantic config (`utils/config_v15.py` + `configs/config_v1_5.yaml`) alongside the untouched v1 config, and a `queue_retry_order` attribute on scheduler classes that lets the environment's fixed queue-retry mechanism run either priority-aware or priority-blind per baseline — resolving an internal ambiguity in the source build prompt (see Task 10 note).

**Tech Stack:** Python 3.11+, Gymnasium, Stable-Baselines3 (PPO), Pydantic v2, FastAPI, pytest, React + TypeScript + Tailwind (frontend dashboard).

## Global Constraints

- `configs/config.yaml` and `utils/helpers.py`'s `AppConfig`/`load_config` stay untouched — v1 reproducibility.
- New v1.5 code goes in the existing top-level packages (`environment/`, `schedulers/`, `metrics/`, `rl/`), never under `backend/environment/`, `backend/schedulers/`, etc. (those are empty compatibility stubs).
- `PriorityTier` values: `gold`, `silver`, `bronze`. Tier weights: gold=3.0, silver=2.0, bronze=1.0.
- Priority distribution: gold=0.15, silver=0.35, bronze=0.50 (must sum to 1.0, validated at config load).
- Deadline ticks by tier: gold=[3,5], silver=[5,10], bronze=[10,20] (uniform sample).
- Reward config starting values: `completion_base=1.0`, `sla_violation_penalty=-2.0`, `overload_penalty_per_unit=-0.5`, `rejection_penalty_base=-1.0`, `queue_penalty_per_job_per_tick=-0.01`, `balance_bonus_weight=0.2`, `starvation_guard={threshold_ticks:15, bonus_per_tick_over:0.05, cap:1.0}`.
- Cluster: 4 servers — A(large, cpu16/mem64), B(medium, cpu8/mem32), C(medium, cpu8/mem32), D(small, cpu4/mem16).
- Observation is exactly 24-dim: 4 servers × 4 fields (`capacity_cpu_norm`, `capacity_mem_norm`, `cpu_util`, `mem_util`) + 1 `queue_length_norm` + 3 job fields + 3 one-hot priority + 1 deadline urgency.
- Action space unchanged: `Discrete(4)`, index into `env.servers` in config order.
- Bronze max-wait bound for acceptance: under 60 ticks (3× Bronze deadline ceiling of 20).
- I (the agent) run all training/evaluation commands myself via Bash and iterate on reward weights against Section 5.4's three failure modes, in order: starvation first, defensive over-rejection second, reward-scale imbalance third.

---

## Task 1: Test scaffold + reward component logging on the existing v1 reward

**Files:**
- Modify: `requirements.txt`
- Create: `tests/__init__.py`
- Create: `tests/test_smoke.py`
- Modify: `environment/cloud_env.py` (existing v1 `step()`, lines 81–131)
- Modify: `rl/train.py` (existing `RichTrainingCallback`/`train_agent`)

**Interfaces:**
- Produces: `CloudResourceEnv.step()` returns `info["reward_components"]: dict[str, float]` with keys `{"schedule", "reject", "completion", "balance", "overload", "queue"}` (v1's existing terms, named) — proves the logging pipeline before the v1.5 reward redesign replaces these exact keys in Task 9.

- [ ] **Step 1: Add pytest to requirements.txt**

Add this line to `requirements.txt` (after `PyYAML>=6.0.1`):

```
pytest>=8.0.0
```

- [ ] **Step 2: Install and create the tests package**

```bash
pip install -r requirements.txt
```

Create `tests/__init__.py`:

```python
"""Test suite for CloudPilot."""
```

Create `tests/test_smoke.py`:

```python
"""Smoke test proving pytest is wired up correctly."""


def test_pytest_runs():
    assert 1 + 1 == 2
```

- [ ] **Step 3: Run the smoke test**

Run: `pytest tests/test_smoke.py -v`
Expected: `1 passed`

- [ ] **Step 4: Break the existing v1 reward into named components**

In `environment/cloud_env.py`, replace the reward-accumulation block inside `step()` (currently lines 92–116, the `reward = 0.0` through `reward -= 5.0` block) with:

```python
        components = {
            "schedule": 0.0,
            "reject": 0.0,
            "completion": 0.0,
            "balance": 0.0,
            "overload": 0.0,
            "queue": 0.0,
        }

        if scheduled:
            self.accepted_jobs += 1
            components["schedule"] += 5.0
        else:
            self.rejected_jobs += 1
            selected_server.queue.append(self.current_job)
            components["reject"] -= 10.0

        self.current_time += 1
        for server in self.servers:
            completed_this_step.extend(server.step(self.current_time))

        self.completed_jobs.extend(completed_this_step)
        components["completion"] += 10.0 * len(completed_this_step)

        if self._is_balanced():
            components["balance"] += 2.0

        if self._has_overloaded_server():
            components["overload"] -= 15.0

        if any(len(server.queue) > 0 for server in self.servers):
            components["queue"] -= 5.0

        reward = sum(components.values())
```

Then update `self.last_step_info` (a few lines below) to include the breakdown:

```python
        self.last_step_info = {
            "scheduled": scheduled,
            "completed_this_step": len(completed_this_step),
            "reward": reward,
            "reward_components": components,
        }
```

- [ ] **Step 5: Write a test proving components sum to the total reward**

Create `tests/test_reward_logging.py`:

```python
"""Proves per-component reward logging matches the total, before the v1.5 redesign."""

from environment.cloud_env import CloudResourceEnv
from utils.helpers import AppConfig


def test_reward_components_sum_to_total():
    config = AppConfig()
    env = CloudResourceEnv(config)
    env.reset(seed=1)

    _, reward, _, _, info = env.step(0)

    assert "reward_components" in info
    assert sum(info["reward_components"].values()) == reward
```

- [ ] **Step 6: Run the test**

Run: `pytest tests/test_reward_logging.py -v`
Expected: `1 passed`

- [ ] **Step 7: Wire per-component TensorBoard logging into rl/train.py**

In `rl/train.py`, add a new callback class above `RichTrainingCallback` and wire `tensorboard_log` on the model:

```python
from stable_baselines3.common.callbacks import CallbackList
```

(add to the existing `from stable_baselines3.common.callbacks import ...` imports)

```python
class RewardComponentCallback(BaseCallback):
    """Log per-component reward means to TensorBoard every step."""

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            components = info.get("reward_components")
            if not components:
                continue
            for name, value in components.items():
                self.logger.record_mean(f"reward_components/{name}", value)
        return True
```

In `train_agent`, change the model construction and `model.learn` call:

```python
    log_dir = results_dir / "training_logs"
    tensorboard_dir = results_dir / "tensorboard"
```

(add `tensorboard_dir` next to the existing `log_dir` assignment)

```python
    model = build_ppo_model(train_env, config.seed, tensorboard_log=tensorboard_dir)
```

```python
    callbacks = CallbackList(
        [RichTrainingCallback(console), RewardComponentCallback(), eval_callback]
    )
    model.learn(
        total_timesteps=config.training_timesteps,
        callback=callbacks,
    )
```

Update `rl/model.py`'s `build_ppo_model` to accept the new parameter:

```python
"""Model factory for PPO scheduling agents."""

from pathlib import Path

from stable_baselines3 import PPO

from environment.cloud_env import CloudResourceEnv


def build_ppo_model(
    env: CloudResourceEnv, seed: int, tensorboard_log: str | Path | None = None
) -> PPO:
    """Create a PPO model with stable-baselines3 defaults."""
    return PPO(
        "MlpPolicy",
        env,
        verbose=0,
        seed=seed,
        tensorboard_log=str(tensorboard_log) if tensorboard_log else None,
    )
```

- [ ] **Step 8: Verify nothing broke with a 2,000-timestep smoke run**

```bash
python -c "from rich.console import Console; from utils.helpers import load_config, ensure_directories; from rl.train import train_agent; c=load_config(); c.training_timesteps=2000; ensure_directories(c); train_agent(c, Console())"
```

Expected: runs to completion with no exceptions, prints per-episode Rich rows, and creates `results/tensorboard/`.

- [ ] **Step 9: Commit**

```bash
git add requirements.txt tests/__init__.py tests/test_smoke.py tests/test_reward_logging.py environment/cloud_env.py rl/train.py rl/model.py
git commit -m "feat: add pytest scaffold and per-component reward logging on v1 reward"
```

---

## Task 2: `environment/priority.py` — tier enum, SLA/starvation logic (no reward yet)

**Files:**
- Create: `environment/priority.py`
- Create: `tests/test_sla_expiry.py`
- Create: `tests/test_starvation_guard.py`

**Interfaces:**
- Produces: `PriorityTier` enum (`GOLD`, `SILVER`, `BRONZE`, values `"gold"/"silver"/"bronze"`), `TIER_ORDER: dict[PriorityTier, int]`, `TickEvents` dataclass (`completions`, `sla_violations`, `rejections`: `list[Job]`), `is_sla_expired(job) -> bool`, `sort_queue_priority_fifo(queue, enqueue_order) -> list[Job]`, `sort_queue_fifo(queue, enqueue_order) -> list[Job]`, `normalized_std_dev(utils: list[float]) -> float`, `starvation_bonus(job, cfg) -> float`.
- Consumes: nothing from earlier tasks (this is the first v1.5-specific module).

- [ ] **Step 1: Write failing tests for SLA expiry and starvation bonus**

Create `tests/test_sla_expiry.py`:

```python
"""SLA expiry: a job that waits past its deadline must be dropped, not placed."""

from environment.priority import PriorityTier, is_sla_expired


class _FakeJob:
    def __init__(self, deadline_ticks: int, ticks_waited: int) -> None:
        self.priority_tier = PriorityTier.GOLD
        self.deadline_ticks = deadline_ticks
        self.ticks_waited = ticks_waited


def test_job_not_expired_before_deadline():
    job = _FakeJob(deadline_ticks=3, ticks_waited=3)
    assert is_sla_expired(job) is False


def test_job_expired_after_deadline():
    job = _FakeJob(deadline_ticks=3, ticks_waited=4)
    assert is_sla_expired(job) is True
```

Create `tests/test_starvation_guard.py`:

```python
"""Starvation guard: bonus stays zero under the threshold, grows and caps above it."""

from types import SimpleNamespace

from environment.priority import PriorityTier, starvation_bonus


class _FakeJob:
    def __init__(self, ticks_waited: int) -> None:
        self.priority_tier = PriorityTier.BRONZE
        self.ticks_waited = ticks_waited


def _guard_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        starvation_guard=SimpleNamespace(threshold_ticks=15, bonus_per_tick_over=0.05, cap=1.0)
    )


def test_no_bonus_under_threshold():
    job = _FakeJob(ticks_waited=15)
    assert starvation_bonus(job, _guard_cfg()) == 0.0


def test_bonus_grows_above_threshold():
    job = _FakeJob(ticks_waited=17)
    assert starvation_bonus(job, _guard_cfg()) == 0.05 * 2


def test_bonus_caps():
    job = _FakeJob(ticks_waited=100)
    assert starvation_bonus(job, _guard_cfg()) == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_sla_expiry.py tests/test_starvation_guard.py -v`
Expected: `ModuleNotFoundError: No module named 'environment.priority'` (or import error)

- [ ] **Step 3: Implement environment/priority.py**

```python
"""Priority tiers, SLA/deadline logic, and starvation guard for CloudPilot v1.5."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from environment.job import Job


class PriorityTier(str, Enum):
    """Job priority tier. Inherits str so it hashes/compares equal to plain tier strings."""

    GOLD = "gold"
    SILVER = "silver"
    BRONZE = "bronze"


TIER_ORDER: dict[PriorityTier, int] = {
    PriorityTier.GOLD: 0,
    PriorityTier.SILVER: 1,
    PriorityTier.BRONZE: 2,
}


@dataclass
class TickEvents:
    """Job outcomes recorded during a single environment tick."""

    completions: list[Job] = field(default_factory=list)
    sla_violations: list[Job] = field(default_factory=list)
    rejections: list[Job] = field(default_factory=list)


def is_sla_expired(job: Job) -> bool:
    """Return True once a queued job has waited strictly past its deadline."""
    return job.ticks_waited > job.deadline_ticks


def sort_queue_priority_fifo(queue: list[Job], enqueue_order: dict[str, int]) -> list[Job]:
    """Sort queued jobs Gold-before-Silver-before-Bronze, FIFO (by enqueue order) within a tier."""
    return sorted(queue, key=lambda job: (TIER_ORDER[job.priority_tier], enqueue_order[job.id]))


def sort_queue_fifo(queue: list[Job], enqueue_order: dict[str, int]) -> list[Job]:
    """Sort queued jobs by arrival order only, ignoring priority tier."""
    return sorted(queue, key=lambda job: enqueue_order[job.id])


def normalized_std_dev(utils: list[float]) -> float:
    """Std dev of utilizations in [0,1], scaled by the max possible std dev (0.5), clipped to [0,1]."""
    if not utils:
        return 0.0
    return float(min(np.std(utils) / 0.5, 1.0))


def starvation_bonus(job: Job, cfg) -> float:
    """Per-tick nudge toward placing a job that has waited past the starvation threshold."""
    guard = cfg.starvation_guard
    if job.ticks_waited <= guard.threshold_ticks:
        return 0.0
    return min(
        guard.bonus_per_tick_over * (job.ticks_waited - guard.threshold_ticks),
        guard.cap,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_sla_expiry.py tests/test_starvation_guard.py -v`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add environment/priority.py tests/test_sla_expiry.py tests/test_starvation_guard.py
git commit -m "feat: add PriorityTier enum, SLA expiry, and starvation guard logic"
```

---

## Task 3: v1.5 config — `utils/config_v15.py` + `configs/config_v1_5.yaml`

**Files:**
- Create: `configs/config_v1_5.yaml`
- Create: `utils/config_v15.py`
- Create: `tests/test_config_v15.py`

**Interfaces:**
- Consumes: `environment.priority.PriorityTier` (Task 2).
- Produces: `AppConfigV15` (pydantic model) with fields `cluster: ClusterConfig`, `priority: PriorityConfig`, `reward: RewardConfig`, `normalization: NormalizationConfig`, `jobs: JobRangeConfig`, `episode_length: int`, `seed: int`, `training_timesteps: int`, `evaluation_episodes: int`, `simulation_speed: float`, `results_dir: str`, `models_dir: str`. `ClusterConfig.servers: list[ServerSpec]` where `ServerSpec` has `id: str`, `cpu_capacity: float`, `mem_capacity: float`. `RewardConfig.tier_weights: dict[PriorityTier, float]`. `PriorityConfig.distribution: dict[PriorityTier, float]`, `.deadline_ticks: dict[PriorityTier, tuple[int, int]]`, `.queue_retry_order: str`. `load_config_v15(path) -> AppConfigV15`. `ensure_directories_v15(config) -> None`.

- [ ] **Step 1: Write the config YAML**

Create `configs/config_v1_5.yaml`:

```yaml
cluster:
  servers:
    - { id: "A", class: "large",  cpu_capacity: 16, mem_capacity: 64 }
    - { id: "B", class: "medium", cpu_capacity: 8,  mem_capacity: 32 }
    - { id: "C", class: "medium", cpu_capacity: 8,  mem_capacity: 32 }
    - { id: "D", class: "small",  cpu_capacity: 4,  mem_capacity: 16 }

jobs:
  min_cpu: 1.0
  max_cpu: 6.0
  min_mem: 1.0
  max_mem: 24.0
  min_duration: 2
  max_duration: 20

priority:
  distribution:
    gold: 0.15
    silver: 0.35
    bronze: 0.50
  deadline_ticks:
    gold:   [3, 5]
    silver: [5, 10]
    bronze: [10, 20]
  queue_retry_order: "priority_fifo"

reward:
  completion_base: 1.0
  tier_weights:
    gold: 3.0
    silver: 2.0
    bronze: 1.0
  sla_violation_penalty: -2.0
  overload_penalty_per_unit: -0.5
  rejection_penalty_base: -1.0
  queue_penalty_per_job_per_tick: -0.01
  balance_bonus_weight: 0.2
  starvation_guard:
    threshold_ticks: 15
    bonus_per_tick_over: 0.05
    cap: 1.0

normalization:
  max_job_cpu: 8
  max_job_mem: 32
  max_runtime: 50
  max_queue_length: 20

episode_length: 500
seed: 42
training_timesteps: 100000
evaluation_episodes: 100
simulation_speed: 2.0
results_dir: "results/v1.5"
models_dir: "models/v1.5"
```

- [ ] **Step 2: Write a failing test for the config loader and validator**

Create `tests/test_config_v15.py`:

```python
"""v1.5 config: loads the real YAML, and rejects a distribution that doesn't sum to 1.0."""

import pytest
from pydantic import ValidationError

from utils.config_v15 import PriorityConfig, load_config_v15


def test_loads_real_config_file():
    config = load_config_v15()
    assert len(config.cluster.servers) == 4
    assert config.cluster.servers[0].id == "A"
    assert config.cluster.servers[0].cpu_capacity == 16


def test_distribution_must_sum_to_one():
    with pytest.raises(ValidationError):
        PriorityConfig(
            distribution={"gold": 0.5, "silver": 0.3, "bronze": 0.3},
            deadline_ticks={"gold": (3, 5), "silver": (5, 10), "bronze": (10, 20)},
        )


def test_distribution_summing_to_one_is_accepted():
    config = PriorityConfig(
        distribution={"gold": 0.15, "silver": 0.35, "bronze": 0.50},
        deadline_ticks={"gold": (3, 5), "silver": (5, 10), "bronze": (10, 20)},
    )
    assert config.queue_retry_order == "priority_fifo"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_config_v15.py -v`
Expected: `ModuleNotFoundError: No module named 'utils.config_v15'`

- [ ] **Step 4: Implement utils/config_v15.py**

```python
"""Nested Pydantic configuration for CloudPilot v1.5 (heterogeneous, multi-objective)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

from environment.priority import PriorityTier


class ServerSpec(BaseModel):
    id: str
    server_class: str = Field(alias="class")
    cpu_capacity: float
    mem_capacity: float

    model_config = {"populate_by_name": True}


class ClusterConfig(BaseModel):
    servers: list[ServerSpec]


class JobRangeConfig(BaseModel):
    min_cpu: float
    max_cpu: float
    min_mem: float
    max_mem: float
    min_duration: int
    max_duration: int


class PriorityConfig(BaseModel):
    distribution: dict[PriorityTier, float]
    deadline_ticks: dict[PriorityTier, tuple[int, int]]
    queue_retry_order: str = "priority_fifo"

    @model_validator(mode="after")
    def distribution_sums_to_one(self) -> "PriorityConfig":
        total = sum(self.distribution.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"priority.distribution must sum to 1.0, got {total}")
        return self


class StarvationGuardConfig(BaseModel):
    threshold_ticks: int
    bonus_per_tick_over: float
    cap: float


class RewardConfig(BaseModel):
    completion_base: float
    tier_weights: dict[PriorityTier, float]
    sla_violation_penalty: float
    overload_penalty_per_unit: float
    rejection_penalty_base: float
    queue_penalty_per_job_per_tick: float
    balance_bonus_weight: float
    starvation_guard: StarvationGuardConfig


class NormalizationConfig(BaseModel):
    max_job_cpu: float
    max_job_mem: float
    max_runtime: float
    max_queue_length: float


class AppConfigV15(BaseModel):
    cluster: ClusterConfig
    jobs: JobRangeConfig
    priority: PriorityConfig
    reward: RewardConfig
    normalization: NormalizationConfig
    episode_length: int = 500
    seed: int = 42
    training_timesteps: int = 100_000
    evaluation_episodes: int = 100
    simulation_speed: float = 2.0
    results_dir: str = "results/v1.5"
    models_dir: str = "models/v1.5"


def load_config_v15(path: str | Path = "configs/config_v1_5.yaml") -> AppConfigV15:
    """Load v1.5 configuration from a YAML file."""
    with Path(path).open("r", encoding="utf-8") as file:
        data: dict[str, Any] = yaml.safe_load(file) or {}
    return AppConfigV15(**data)


def ensure_directories_v15(config: AppConfigV15) -> None:
    """Create output directories required by v1.5 training/evaluation."""
    Path(config.results_dir).mkdir(parents=True, exist_ok=True)
    Path(config.models_dir).mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_config_v15.py -v`
Expected: `3 passed`

- [ ] **Step 6: Commit**

```bash
git add configs/config_v1_5.yaml utils/config_v15.py tests/test_config_v15.py
git commit -m "feat: add nested v1.5 config schema with distribution validator"
```

---

## Task 4: `compute_reward` — the full multi-term reward function

**Files:**
- Modify: `environment/priority.py` (append `compute_reward`)
- Create: `tests/test_reward.py`

**Interfaces:**
- Consumes: `TickEvents`, `PriorityTier`, `normalized_std_dev`, `starvation_bonus` (Task 2); `RewardConfig`, `StarvationGuardConfig` (Task 3, for typing/construction in the test).
- Produces: `compute_reward(tick_events: TickEvents, servers: list[Server], queue: list[Job], cfg: RewardConfig) -> tuple[float, dict[str, float]]`. The returned dict always has exactly these keys: `completion`, `sla_violation`, `rejection`, `overload`, `queue`, `balance`, `starvation_bonus`.

- [ ] **Step 1: Write the hand-calculated reward test (the spec's single most valuable test)**

Create `tests/test_reward.py`:

```python
"""Reward calculation: one completion per tier, one SLA violation, one rejection, known overload."""

from environment.priority import PriorityTier, TickEvents, compute_reward
from utils.config_v15 import RewardConfig, StarvationGuardConfig


class _FakeJob:
    def __init__(self, tier: PriorityTier, ticks_waited: int = 0) -> None:
        self.priority_tier = tier
        self.ticks_waited = ticks_waited


class _FakeServer:
    def __init__(self, capacity: dict[str, float], used: dict[str, float]) -> None:
        self.capacity = capacity
        self.used = used

    def utilization(self, resource: str) -> float:
        return self.used[resource] / self.capacity[resource]


def _cfg() -> RewardConfig:
    return RewardConfig(
        completion_base=1.0,
        tier_weights={"gold": 3.0, "silver": 2.0, "bronze": 1.0},
        sla_violation_penalty=-2.0,
        overload_penalty_per_unit=-0.5,
        rejection_penalty_base=-1.0,
        queue_penalty_per_job_per_tick=-0.01,
        balance_bonus_weight=0.2,
        starvation_guard=StarvationGuardConfig(threshold_ticks=15, bonus_per_tick_over=0.05, cap=1.0),
    )


def test_reward_matches_hand_calculated_value():
    cfg = _cfg()
    tick_events = TickEvents(
        completions=[
            _FakeJob(PriorityTier.GOLD),
            _FakeJob(PriorityTier.SILVER),
            _FakeJob(PriorityTier.BRONZE),
        ],
        sla_violations=[_FakeJob(PriorityTier.SILVER)],
        rejections=[_FakeJob(PriorityTier.BRONZE)],
    )
    servers = [
        _FakeServer(capacity={"cpu": 10.0, "mem": 10.0}, used={"cpu": 12.0, "mem": 5.0}),
        _FakeServer(capacity={"cpu": 10.0, "mem": 10.0}, used={"cpu": 5.0, "mem": 5.0}),
    ]
    queue: list = []

    total, components = compute_reward(tick_events, servers, queue, cfg)

    expected_completion = 1.0 * (3.0 + 2.0 + 1.0)
    expected_sla_violation = -2.0 * 2.0
    expected_rejection = -1.0 * 1.0
    expected_overload = -0.5 * 2.0
    expected_queue = -0.01 * 0
    utils = [0.5, 1.2, 0.5, 0.5]
    expected_balance = 0.2 * (1 - min((sum((u - sum(utils) / 4) ** 2 for u in utils) / 4) ** 0.5 / 0.5, 1.0))

    assert components["completion"] == expected_completion
    assert components["sla_violation"] == expected_sla_violation
    assert components["rejection"] == expected_rejection
    assert components["overload"] == expected_overload
    assert components["queue"] == expected_queue
    assert components["balance"] == pytest.approx(expected_balance)
    assert components["starvation_bonus"] == 0.0
    assert total == pytest.approx(sum(components.values()))


def test_starvation_bonus_included_for_waiting_queue_jobs():
    cfg = _cfg()
    tick_events = TickEvents()
    servers = [_FakeServer(capacity={"cpu": 10.0}, used={"cpu": 0.0})]
    queue = [_FakeJob(PriorityTier.BRONZE, ticks_waited=17)]

    _, components = compute_reward(tick_events, servers, queue, cfg)

    assert components["starvation_bonus"] == pytest.approx(0.05 * 2)
```

Add `import pytest` at the top of the file.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_reward.py -v`
Expected: `ImportError: cannot import name 'compute_reward' from 'environment.priority'`

- [ ] **Step 3: Implement compute_reward, appended to environment/priority.py**

Append to `environment/priority.py`:

```python
def compute_reward(
    tick_events: TickEvents,
    servers,
    queue: list[Job],
    cfg,
) -> tuple[float, dict[str, float]]:
    """Compute total reward and its named components for one tick."""
    components = {
        "completion": 0.0,
        "sla_violation": 0.0,
        "rejection": 0.0,
        "overload": 0.0,
        "queue": 0.0,
        "balance": 0.0,
        "starvation_bonus": 0.0,
    }

    for job in tick_events.completions:
        components["completion"] += cfg.completion_base * cfg.tier_weights[job.priority_tier]

    for job in tick_events.sla_violations:
        components["sla_violation"] += cfg.sla_violation_penalty * cfg.tier_weights[job.priority_tier]

    for job in tick_events.rejections:
        components["rejection"] += cfg.rejection_penalty_base * cfg.tier_weights[job.priority_tier]

    total_overload_units = sum(
        max(0.0, server.used[resource] - server.capacity[resource])
        for server in servers
        for resource in server.capacity
    )
    components["overload"] = cfg.overload_penalty_per_unit * total_overload_units

    components["queue"] = cfg.queue_penalty_per_job_per_tick * len(queue)

    utils = [server.utilization(resource) for server in servers for resource in server.capacity]
    components["balance"] = cfg.balance_bonus_weight * (1 - normalized_std_dev(utils))

    for job in queue:
        components["starvation_bonus"] += starvation_bonus(job, cfg)

    total = sum(components.values())
    return total, components
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_reward.py -v`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add environment/priority.py tests/test_reward.py
git commit -m "feat: implement multi-term compute_reward with hand-calculated test"
```

---

## Task 5: `environment/job.py` — dict-based Job with priority/deadline fields

**Files:**
- Modify: `environment/job.py` (full rewrite)
- Create: `tests/test_job.py`

**Interfaces:**
- Consumes: `PriorityTier` (Task 2).
- Produces: `Job` dataclass — `id: str`, `requirements: dict[str, float]`, `runtime: int`, `arrival_tick: int`, `priority_tier: PriorityTier`, `deadline_ticks: int`, `ticks_waited: int = 0`, `remaining_time: int` (auto-set from `runtime`), `start_tick: int | None = None`, `completion_tick: int | None = None`, `.response_time` property.

- [ ] **Step 1: Write the failing test**

Create `tests/test_job.py`:

```python
"""Job dataclass: dict-based requirements, priority/deadline fields, response_time."""

from environment.job import Job
from environment.priority import PriorityTier


def _make_job(**overrides) -> Job:
    defaults = dict(
        id="job-0",
        requirements={"cpu": 2.0, "mem": 4.0},
        runtime=5,
        arrival_tick=0,
        priority_tier=PriorityTier.GOLD,
        deadline_ticks=3,
    )
    defaults.update(overrides)
    return Job(**defaults)


def test_remaining_time_starts_at_runtime():
    job = _make_job(runtime=7)
    assert job.remaining_time == 7


def test_response_time_none_until_completed():
    job = _make_job()
    assert job.response_time is None


def test_response_time_after_completion():
    job = _make_job(arrival_tick=2)
    job.completion_tick = 10
    assert job.response_time == 8


def test_ticks_waited_defaults_to_zero():
    job = _make_job()
    assert job.ticks_waited == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_job.py -v`
Expected: `TypeError` (old `Job` fields don't match: `job_id`, `cpu_required`, etc.)

- [ ] **Step 3: Rewrite environment/job.py**

```python
"""Job model used by the CloudPilot v1.5 simulator (dict-based, priority-aware)."""

from __future__ import annotations

from dataclasses import dataclass, field

from environment.priority import PriorityTier


@dataclass(slots=True)
class Job:
    """A simulated cloud job with dict-based resource requirements and a priority tier."""

    id: str
    requirements: dict[str, float]
    runtime: int
    arrival_tick: int
    priority_tier: PriorityTier
    deadline_ticks: int
    ticks_waited: int = 0
    remaining_time: int = field(init=False)
    start_tick: int | None = None
    completion_tick: int | None = None

    def __post_init__(self) -> None:
        self.remaining_time = self.runtime

    @property
    def response_time(self) -> int | None:
        """Return completion tick minus arrival tick once complete."""
        if self.completion_tick is None:
            return None
        return self.completion_tick - self.arrival_tick
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_job.py -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add environment/job.py tests/test_job.py
git commit -m "feat: rewrite Job as dict-based dataclass with priority tier and deadline"
```

---

## Task 6: `environment/server.py` — dict-based Server (capacity/used)

**Files:**
- Modify: `environment/server.py` (full rewrite)
- Create: `tests/test_server.py`

**Interfaces:**
- Consumes: `Job` (Task 5).
- Produces: `Server` dataclass — `id: str`, `capacity: dict[str, float]`, `used: dict[str, float]` (defaults to zeros matching `capacity`'s keys), `running_jobs: list[Job]`. Methods: `.reset()`, `.utilization(resource: str) -> float`, `.average_utilization` (property), `.can_run(job) -> bool`, `.accept_job(job, current_tick) -> bool`, `.step(current_tick) -> list[Job]` (returns newly completed jobs). No `.queue` field — the queue is global (owned by `CloudResourceEnv`, Task 9).

- [ ] **Step 1: Write the failing test**

Create `tests/test_server.py`:

```python
"""Server dataclass: dict-based capacity/used, fit checks, accept/step lifecycle."""

from environment.job import Job
from environment.priority import PriorityTier
from environment.server import Server


def _make_job(cpu: float, mem: float, runtime: int = 3) -> Job:
    return Job(
        id="job-0",
        requirements={"cpu": cpu, "mem": mem},
        runtime=runtime,
        arrival_tick=0,
        priority_tier=PriorityTier.GOLD,
        deadline_ticks=3,
    )


def test_used_defaults_to_zero_per_resource():
    server = Server(id="A", capacity={"cpu": 16.0, "mem": 64.0})
    assert server.used == {"cpu": 0.0, "mem": 0.0}


def test_can_run_true_when_fits():
    server = Server(id="A", capacity={"cpu": 16.0, "mem": 64.0})
    assert server.can_run(_make_job(cpu=4.0, mem=8.0)) is True


def test_can_run_false_when_any_resource_overflows():
    server = Server(id="D", capacity={"cpu": 4.0, "mem": 16.0})
    assert server.can_run(_make_job(cpu=5.0, mem=8.0)) is False


def test_accept_job_updates_usage_and_running_jobs():
    server = Server(id="A", capacity={"cpu": 16.0, "mem": 64.0})
    job = _make_job(cpu=4.0, mem=8.0)

    accepted = server.accept_job(job, current_tick=1)

    assert accepted is True
    assert server.used == {"cpu": 4.0, "mem": 8.0}
    assert job in server.running_jobs
    assert job.start_tick == 1


def test_accept_job_rejected_when_infeasible():
    server = Server(id="D", capacity={"cpu": 4.0, "mem": 16.0})
    job = _make_job(cpu=5.0, mem=8.0)

    assert server.accept_job(job, current_tick=1) is False
    assert job not in server.running_jobs


def test_step_completes_job_when_runtime_elapses():
    server = Server(id="A", capacity={"cpu": 16.0, "mem": 64.0})
    job = _make_job(cpu=4.0, mem=8.0, runtime=2)
    server.accept_job(job, current_tick=0)

    completed_tick_1 = server.step(current_tick=1)
    assert completed_tick_1 == []

    completed_tick_2 = server.step(current_tick=2)
    assert completed_tick_2 == [job]
    assert job.completion_tick == 2
    assert server.used == {"cpu": 0.0, "mem": 0.0}


def test_average_utilization():
    server = Server(id="A", capacity={"cpu": 10.0, "mem": 10.0})
    server.used = {"cpu": 5.0, "mem": 2.0}
    assert server.average_utilization == 0.35


def test_reset_clears_usage_and_running_jobs():
    server = Server(id="A", capacity={"cpu": 16.0, "mem": 64.0})
    server.accept_job(_make_job(cpu=4.0, mem=8.0), current_tick=0)

    server.reset()

    assert server.used == {"cpu": 0.0, "mem": 0.0}
    assert server.running_jobs == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_server.py -v`
Expected: `TypeError` (old `Server` uses `cpu_capacity`/`memory_capacity`, not `capacity` dict)

- [ ] **Step 3: Rewrite environment/server.py**

```python
"""Server model for the CloudPilot v1.5 simulator (dict-based resource capacity)."""

from __future__ import annotations

from dataclasses import dataclass, field

from environment.job import Job


@dataclass(slots=True)
class Server:
    """A server with per-resource capacity, expressed as a resource dict."""

    id: str
    capacity: dict[str, float]
    used: dict[str, float] = field(default_factory=dict)
    running_jobs: list[Job] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.used:
            self.used = {resource: 0.0 for resource in self.capacity}

    def reset(self) -> None:
        """Clear all usage and running jobs."""
        self.used = {resource: 0.0 for resource in self.capacity}
        self.running_jobs.clear()

    def utilization(self, resource: str) -> float:
        """Return utilization in [0, 1] for one resource."""
        return self.used[resource] / self.capacity[resource]

    @property
    def average_utilization(self) -> float:
        """Mean utilization across all resources."""
        return sum(self.utilization(resource) for resource in self.capacity) / len(self.capacity)

    def can_run(self, job: Job) -> bool:
        """Return True when every required resource fits in current free capacity."""
        return all(
            self.used[resource] + amount <= self.capacity[resource]
            for resource, amount in job.requirements.items()
        )

    def accept_job(self, job: Job, current_tick: int) -> bool:
        """Start a job immediately when capacity is available."""
        if not self.can_run(job):
            return False

        job.start_tick = current_tick
        for resource, amount in job.requirements.items():
            self.used[resource] += amount
        self.running_jobs.append(job)
        return True

    def step(self, current_tick: int) -> list[Job]:
        """Advance running jobs by one tick and release completed jobs."""
        completed_jobs: list[Job] = []
        still_running: list[Job] = []

        for job in self.running_jobs:
            job.remaining_time -= 1
            if job.remaining_time <= 0:
                job.completion_tick = current_tick
                for resource, amount in job.requirements.items():
                    self.used[resource] -= amount
                completed_jobs.append(job)
            else:
                still_running.append(job)

        self.running_jobs = still_running
        return completed_jobs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_server.py -v`
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add environment/server.py tests/test_server.py
git commit -m "feat: rewrite Server as dict-based dataclass, drop dead per-server queue"
```

---

## Task 7: `environment/heterogeneous.py` — build servers from config

**Files:**
- Create: `environment/heterogeneous.py`
- Create: `tests/test_heterogeneous.py`

**Interfaces:**
- Consumes: `Server` (Task 6), `ClusterConfig`/`ServerSpec` (Task 3).
- Produces: `build_servers_from_config(cluster: ClusterConfig) -> list[Server]`, `max_capacity_per_resource(cluster: ClusterConfig) -> dict[str, float]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_heterogeneous.py`:

```python
"""Heterogeneous server construction from cluster config."""

from environment.heterogeneous import build_servers_from_config, max_capacity_per_resource
from utils.config_v15 import load_config_v15


def test_builds_four_servers_in_config_order():
    config = load_config_v15()
    servers = build_servers_from_config(config.cluster)

    assert [s.id for s in servers] == ["A", "B", "C", "D"]
    assert servers[0].capacity == {"cpu": 16.0, "mem": 64.0}
    assert servers[3].capacity == {"cpu": 4.0, "mem": 16.0}
    assert servers[0].used == {"cpu": 0.0, "mem": 0.0}


def test_max_capacity_per_resource():
    config = load_config_v15()
    maxima = max_capacity_per_resource(config.cluster)

    assert maxima == {"cpu": 16.0, "mem": 64.0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_heterogeneous.py -v`
Expected: `ModuleNotFoundError: No module named 'environment.heterogeneous'`

- [ ] **Step 3: Implement environment/heterogeneous.py**

```python
"""Build heterogeneous Server instances from cluster config."""

from __future__ import annotations

from environment.server import Server
from utils.config_v15 import ClusterConfig


def build_servers_from_config(cluster: ClusterConfig) -> list[Server]:
    """Create one Server per configured spec, in config order."""
    return [
        Server(id=spec.id, capacity={"cpu": spec.cpu_capacity, "mem": spec.mem_capacity})
        for spec in cluster.servers
    ]


def max_capacity_per_resource(cluster: ClusterConfig) -> dict[str, float]:
    """Return the maximum configured capacity per resource, for observation normalization."""
    return {
        "cpu": max(spec.cpu_capacity for spec in cluster.servers),
        "mem": max(spec.mem_capacity for spec in cluster.servers),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_heterogeneous.py -v`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add environment/heterogeneous.py tests/test_heterogeneous.py
git commit -m "feat: add heterogeneous server factory built from cluster config"
```

---

## Task 8: `environment/generator.py` — priority-aware job generator

**Files:**
- Modify: `environment/generator.py` (full rewrite)
- Create: `tests/test_priority_sampling.py`

**Interfaces:**
- Consumes: `Job` (Task 5), `PriorityTier` (Task 2), `PriorityConfig` (Task 3).
- Produces: `JobGenerator(min_cpu, max_cpu, min_mem, max_mem, min_duration, max_duration, priority_cfg, seed=None)`, `.reset(seed=None)`, `.generate(arrival_tick) -> Job`.

- [ ] **Step 1: Write the failing test (spec's exact 10,000-job distribution test)**

Create `tests/test_priority_sampling.py`:

```python
"""Priority sampling: 10,000 jobs with a fixed seed match the configured tier distribution."""

from collections import Counter

from environment.generator import JobGenerator
from utils.config_v15 import load_config_v15


def test_tier_proportions_within_tolerance():
    config = load_config_v15()
    generator = JobGenerator(
        min_cpu=config.jobs.min_cpu,
        max_cpu=config.jobs.max_cpu,
        min_mem=config.jobs.min_mem,
        max_mem=config.jobs.max_mem,
        min_duration=config.jobs.min_duration,
        max_duration=config.jobs.max_duration,
        priority_cfg=config.priority,
        seed=42,
    )

    jobs = [generator.generate(arrival_tick=i) for i in range(10_000)]
    counts = Counter(job.priority_tier for job in jobs)

    for tier, expected_probability in config.priority.distribution.items():
        observed = counts[tier] / 10_000
        assert abs(observed - expected_probability) < 0.02


def test_deadline_ticks_within_configured_range():
    config = load_config_v15()
    generator = JobGenerator(
        min_cpu=config.jobs.min_cpu,
        max_cpu=config.jobs.max_cpu,
        min_mem=config.jobs.min_mem,
        max_mem=config.jobs.max_mem,
        min_duration=config.jobs.min_duration,
        max_duration=config.jobs.max_duration,
        priority_cfg=config.priority,
        seed=7,
    )

    for i in range(2_000):
        job = generator.generate(arrival_tick=i)
        low, high = config.priority.deadline_ticks[job.priority_tier]
        assert low <= job.deadline_ticks <= high


def test_seeded_generation_is_reproducible():
    config = load_config_v15()

    def _first_five(seed: int) -> list[str]:
        generator = JobGenerator(
            min_cpu=config.jobs.min_cpu,
            max_cpu=config.jobs.max_cpu,
            min_mem=config.jobs.min_mem,
            max_mem=config.jobs.max_mem,
            min_duration=config.jobs.min_duration,
            max_duration=config.jobs.max_duration,
            priority_cfg=config.priority,
            seed=seed,
        )
        return [generator.generate(i).priority_tier.value for i in range(5)]

    assert _first_five(42) == _first_five(42)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_priority_sampling.py -v`
Expected: `TypeError` (old `JobGenerator.__init__` signature doesn't accept `priority_cfg`)

- [ ] **Step 3: Rewrite environment/generator.py**

```python
"""Priority-aware random job generator for CloudPilot v1.5."""

from __future__ import annotations

import numpy as np

from environment.job import Job
from utils.config_v15 import PriorityConfig


class JobGenerator:
    """Generate one random job per simulation tick, with a sampled priority tier and SLA deadline."""

    def __init__(
        self,
        min_cpu: float,
        max_cpu: float,
        min_mem: float,
        max_mem: float,
        min_duration: int,
        max_duration: int,
        priority_cfg: PriorityConfig,
        seed: int | None = None,
    ) -> None:
        self.min_cpu = min_cpu
        self.max_cpu = max_cpu
        self.min_mem = min_mem
        self.max_mem = max_mem
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.priority_cfg = priority_cfg
        self.rng = np.random.default_rng(seed)
        self.next_job_id = 0

    def reset(self, seed: int | None = None) -> None:
        """Reset sequence state and optionally reseed randomness."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.next_job_id = 0

    def generate(self, arrival_tick: int) -> Job:
        """Create a new job for the current simulation tick, part of the same seeded RNG stream."""
        tiers = list(self.priority_cfg.distribution.keys())
        probabilities = [self.priority_cfg.distribution[tier] for tier in tiers]
        tier = tiers[self.rng.choice(len(tiers), p=probabilities)]

        low, high = self.priority_cfg.deadline_ticks[tier]
        deadline_ticks = int(self.rng.integers(low, high + 1))

        job = Job(
            id=f"job-{self.next_job_id}",
            requirements={
                "cpu": float(self.rng.uniform(self.min_cpu, self.max_cpu)),
                "mem": float(self.rng.uniform(self.min_mem, self.max_mem)),
            },
            runtime=int(self.rng.integers(self.min_duration, self.max_duration + 1)),
            arrival_tick=arrival_tick,
            priority_tier=tier,
            deadline_ticks=deadline_ticks,
        )
        self.next_job_id += 1
        return job
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_priority_sampling.py -v`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add environment/generator.py tests/test_priority_sampling.py
git commit -m "feat: sample priority tier and SLA deadline in the seeded job generator"
```

---

## Task 9: `environment/cloud_env.py` — full v1.5 rewrite (24-dim obs, global queue, reward wiring)

**Files:**
- Modify: `environment/cloud_env.py` (full rewrite of the class body from Task 1's state)
- Create: `tests/test_cloud_env.py`

**Interfaces:**
- Consumes: `AppConfigV15` (Task 3), `Job`/`PriorityTier` (Tasks 2, 5), `Server` (Task 6), `build_servers_from_config`/`max_capacity_per_resource` (Task 7), `JobGenerator` (Task 8), `TickEvents`/`is_sla_expired`/`sort_queue_priority_fifo`/`sort_queue_fifo`/`compute_reward` (Tasks 2, 4).
- Produces: `CloudResourceEnv(config: AppConfigV15, queue_retry_order: str | None = None)`. `self.servers: list[Server]`, `self.queue: list[Job]`, `self.current_job: Job | None`, `self.current_tick: int`, `self.queue_retry_order: str` (mutable, read by `LiveSimulation` in Task 16). `observation_space`: `Box(shape=(24,))`. `action_space`: `Discrete(len(servers))`. `step(action) -> (obs, reward, terminated, truncated, info)` where `info` includes `scheduled`, `completed_this_step`, `reward`, `reward_components`, `accepted_jobs`, `completed_jobs`, `rejected_jobs`, `sla_violations`, `priority_weighted_completion_rate`, `max_wait_by_tier`, `average_cpu_utilization`, `average_memory_utilization`, `average_queue_length`, `average_response_time`, `episode_reward`, `reward_components_total`.

- [ ] **Step 1: Write failing tests covering queue retry, infeasible-action semantics, and observation shape**

Create `tests/test_cloud_env.py`:

```python
"""CloudResourceEnv v1.5: 24-dim observation, global queue retry, infeasible-action semantics."""

import numpy as np

from environment.cloud_env import CloudResourceEnv
from environment.priority import PriorityTier
from utils.config_v15 import load_config_v15


def _tiny_config():
    config = load_config_v15()
    config.episode_length = 50
    return config


def test_observation_is_24_dimensional():
    env = CloudResourceEnv(_tiny_config())
    obs, _ = env.reset(seed=1)

    assert env.observation_space.shape == (24,)
    assert obs.shape == (24,)
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0)


def test_action_space_is_discrete_four():
    env = CloudResourceEnv(_tiny_config())
    assert env.action_space.n == 4


def test_infeasible_action_queues_job_that_fits_elsewhere():
    env = CloudResourceEnv(_tiny_config())
    env.reset(seed=1)
    # Fill server 0 (A) to capacity so it can't take anything else.
    env.servers[0].used = dict(env.servers[0].capacity)

    env.current_job = env.job_generator.generate(env.current_tick)
    env.current_job.requirements = {"cpu": 1.0, "mem": 1.0}
    _, _, _, _, info = env.step(0)

    assert info["scheduled"] is False
    assert len(env.queue) == 1
    assert info["rejected_jobs"] == 0


def test_job_larger_than_every_server_is_rejected_not_queued():
    env = CloudResourceEnv(_tiny_config())
    env.reset(seed=1)
    env.current_job = env.job_generator.generate(env.current_tick)
    env.current_job.requirements = {"cpu": 999.0, "mem": 999.0}

    _, _, _, _, info = env.step(0)

    assert info["scheduled"] is False
    assert len(env.queue) == 0
    assert info["rejected_jobs"] == 1


def test_sla_expired_job_counted_as_both_violation_and_rejection():
    env = CloudResourceEnv(_tiny_config())
    env.reset(seed=1)
    stuck_job = env.job_generator.generate(env.current_tick)
    stuck_job.priority_tier = PriorityTier.GOLD
    stuck_job.deadline_ticks = 1
    stuck_job.requirements = {"cpu": 999.0, "mem": 999.0}
    env.queue = [stuck_job]
    env._enqueue_order[stuck_job.id] = 0
    env._enqueue_counter = 1

    for _ in range(3):
        env.current_job.requirements = {"cpu": 999.0, "mem": 999.0}
        _, _, _, _, info = env.step(0)
        if stuck_job.id not in [j.id for j in env.queue]:
            break

    assert info["sla_violations"] >= 1


def test_priority_fifo_retry_places_gold_before_bronze():
    env = CloudResourceEnv(_tiny_config(), queue_retry_order="priority_fifo")
    env.reset(seed=1)
    for server in env.servers:
        server.used = dict(server.capacity)
    # Only server D (index 3, small: cpu4/mem16) has free capacity, and only enough
    # room for ONE of the two queued jobs below — this makes priority order observable.
    env.servers[3].used = {"cpu": 0.0, "mem": 0.0}

    bronze_job = env.job_generator.generate(0)
    bronze_job.priority_tier = PriorityTier.BRONZE
    bronze_job.requirements = {"cpu": 3.0, "mem": 10.0}
    gold_job = env.job_generator.generate(0)
    gold_job.priority_tier = PriorityTier.GOLD
    gold_job.requirements = {"cpu": 3.0, "mem": 10.0}

    env.queue = [bronze_job, gold_job]
    env._enqueue_order = {bronze_job.id: 0, gold_job.id: 1}
    env._enqueue_counter = 2

    env.current_job.requirements = {"cpu": 999.0, "mem": 999.0}
    env.step(0)

    assert gold_job in env.servers[3].running_jobs
    assert bronze_job not in env.servers[3].running_jobs
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cloud_env.py -v`
Expected: failures — old `CloudResourceEnv.__init__` still takes `AppConfig`, not `AppConfigV15`, and has no `queue`/`queue_retry_order`.

- [ ] **Step 3: Rewrite environment/cloud_env.py**

```python
"""Gymnasium environment for CloudPilot v1.5 (heterogeneous, priority-aware, multi-objective)."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environment.generator import JobGenerator
from environment.heterogeneous import build_servers_from_config, max_capacity_per_resource
from environment.job import Job
from environment.priority import (
    PriorityTier,
    TickEvents,
    compute_reward,
    is_sla_expired,
    sort_queue_fifo,
    sort_queue_priority_fifo,
)
from environment.server import Server
from utils.config_v15 import AppConfigV15


class CloudResourceEnv(gym.Env[np.ndarray, int]):
    """Heterogeneous, priority-aware cloud scheduling environment."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: AppConfigV15, queue_retry_order: str | None = None) -> None:
        super().__init__()
        self.config = config
        self.queue_retry_order = queue_retry_order or config.priority.queue_retry_order
        self.max_capacity = max_capacity_per_resource(config.cluster)
        self.servers: list[Server] = build_servers_from_config(config.cluster)
        self.job_generator = JobGenerator(
            min_cpu=config.jobs.min_cpu,
            max_cpu=config.jobs.max_cpu,
            min_mem=config.jobs.min_mem,
            max_mem=config.jobs.max_mem,
            min_duration=config.jobs.min_duration,
            max_duration=config.jobs.max_duration,
            priority_cfg=config.priority,
            seed=config.seed,
        )

        self.queue: list[Job] = []
        self._enqueue_order: dict[str, int] = {}
        self._enqueue_counter = 0

        num_servers = len(self.servers)
        observation_size = num_servers * 4 + 1 + 3 + 3 + 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(observation_size,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(num_servers)

        self.current_tick = 0
        self.current_job: Job | None = None
        self.completed_jobs: list[Job] = []
        self.rejected_jobs = 0
        self.sla_violation_count = 0
        self.accepted_jobs = 0
        self.episode_reward = 0.0
        self.reward_components_total: dict[str, float] = {}
        self.completed_tier_weight = 0.0
        self.total_tier_weight_seen = 0.0
        self.max_wait_by_tier: dict[str, int] = {tier.value: 0 for tier in PriorityTier}
        self.last_step_info: dict[str, Any] = {}

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset simulator state and return the initial observation."""
        super().reset(seed=seed)
        actual_seed = self.config.seed if seed is None else seed
        self.job_generator.reset(actual_seed)

        for server in self.servers:
            server.reset()

        self.queue = []
        self._enqueue_order = {}
        self._enqueue_counter = 0
        self.current_tick = 0
        self.completed_jobs = []
        self.rejected_jobs = 0
        self.sla_violation_count = 0
        self.accepted_jobs = 0
        self.episode_reward = 0.0
        self.reward_components_total = {}
        self.completed_tier_weight = 0.0
        self.total_tier_weight_seen = 0.0
        self.max_wait_by_tier = {tier.value: 0 for tier in PriorityTier}
        self.last_step_info = {}

        self.current_job = self.job_generator.generate(self.current_tick)
        self.total_tier_weight_seen += self.config.reward.tier_weights[self.current_job.priority_tier]

        return self._get_observation(), self._get_info()

    def step(
        self, action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Age the queue, drop SLA-expired jobs, retry the queue, place the incoming job."""
        if self.current_job is None:
            raise RuntimeError("Environment must be reset before stepping.")

        tick_events = TickEvents()

        # Step 1: age every queued job by one tick.
        for job in self.queue:
            job.ticks_waited += 1
            tier = job.priority_tier.value
            self.max_wait_by_tier[tier] = max(self.max_wait_by_tier[tier], job.ticks_waited)

        # Step 2: drop SLA-expired jobs; count as both violation and rejection.
        still_queued: list[Job] = []
        for job in self.queue:
            if is_sla_expired(job):
                tick_events.sla_violations.append(job)
                tick_events.rejections.append(job)
                self.sla_violation_count += 1
                self.rejected_jobs += 1
            else:
                still_queued.append(job)
        self.queue = still_queued

        # Step 3: retry remaining queued jobs against free capacity, priority-then-FIFO (or plain FIFO).
        ordered_queue = self._sort_queue(self.queue)
        still_queued = []
        for job in ordered_queue:
            fitting = [server for server in self.servers if server.can_run(job)]
            if fitting:
                target = min(fitting, key=lambda server: server.average_utilization)
                target.accept_job(job, self.current_tick)
            else:
                still_queued.append(job)
        self.queue = still_queued

        # Step 4: place the newly arrived job (the only agent-controlled decision).
        incoming = self.current_job
        selected_server = self.servers[int(action)]
        scheduled = selected_server.accept_job(incoming, self.current_tick)
        if scheduled:
            self.accepted_jobs += 1
        else:
            fits_anywhere = any(server.can_run(incoming) for server in self.servers)
            if fits_anywhere:
                self._enqueue(incoming)
            else:
                tick_events.rejections.append(incoming)
                self.rejected_jobs += 1

        self.current_tick += 1
        for server in self.servers:
            tick_events.completions.extend(server.step(self.current_tick))
        self.completed_jobs.extend(tick_events.completions)
        self.completed_tier_weight += sum(
            self.config.reward.tier_weights[job.priority_tier] for job in tick_events.completions
        )

        reward, components = compute_reward(tick_events, self.servers, self.queue, self.config.reward)
        self.episode_reward += reward
        for name, value in components.items():
            self.reward_components_total[name] = self.reward_components_total.get(name, 0.0) + value

        terminated = self.current_tick >= self.config.episode_length
        truncated = False
        self.current_job = None if terminated else self.job_generator.generate(self.current_tick)
        if self.current_job is not None:
            self.total_tier_weight_seen += self.config.reward.tier_weights[self.current_job.priority_tier]

        self.last_step_info = {
            "scheduled": scheduled,
            "completed_this_step": len(tick_events.completions),
            "reward": reward,
            "reward_components": components,
        }
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def render(self) -> None:
        """Print a compact summary of the current cluster state."""
        utilization = [
            f"{server.id}: CPU {server.utilization('cpu'):.2f}, MEM {server.utilization('mem'):.2f}"
            for server in self.servers
        ]
        print(f"t={self.current_tick} | queue={len(self.queue)} | " + " | ".join(utilization))

    def _enqueue(self, job: Job) -> None:
        self.queue.append(job)
        self._enqueue_order[job.id] = self._enqueue_counter
        self._enqueue_counter += 1

    def _sort_queue(self, queue: list[Job]) -> list[Job]:
        if self.queue_retry_order == "priority_fifo":
            return sort_queue_priority_fifo(queue, self._enqueue_order)
        return sort_queue_fifo(queue, self._enqueue_order)

    def _get_observation(self) -> np.ndarray:
        server_state: list[float] = []
        for server in self.servers:
            server_state.extend([
                server.capacity["cpu"] / self.max_capacity["cpu"],
                server.capacity["mem"] / self.max_capacity["mem"],
                server.utilization("cpu"),
                server.utilization("mem"),
            ])

        queue_norm = min(len(self.queue) / self.config.normalization.max_queue_length, 1.0)

        if self.current_job is None:
            job_state = [0.0, 0.0, 0.0]
            priority_state = [0.0, 0.0, 0.0]
            urgency_state = [0.0]
        else:
            job = self.current_job
            job_state = [
                job.requirements["cpu"] / self.config.normalization.max_job_cpu,
                job.requirements["mem"] / self.config.normalization.max_job_mem,
                job.runtime / self.config.normalization.max_runtime,
            ]
            priority_state = [
                1.0 if job.priority_tier == PriorityTier.GOLD else 0.0,
                1.0 if job.priority_tier == PriorityTier.SILVER else 0.0,
                1.0 if job.priority_tier == PriorityTier.BRONZE else 0.0,
            ]
            max_deadline = max(high for _, high in self.config.priority.deadline_ticks.values())
            urgency_state = [job.deadline_ticks / max_deadline]

        return np.asarray(
            server_state + [queue_norm] + job_state + priority_state + urgency_state,
            dtype=np.float32,
        )

    def _get_info(self) -> dict[str, Any]:
        avg_cpu = float(np.mean([server.utilization("cpu") for server in self.servers]))
        avg_mem = float(np.mean([server.utilization("mem") for server in self.servers]))
        response_times = [
            job.response_time for job in self.completed_jobs if job.response_time is not None
        ]
        avg_response_time = float(np.mean(response_times)) if response_times else 0.0
        completion_rate = (
            self.completed_tier_weight / self.total_tier_weight_seen
            if self.total_tier_weight_seen
            else 0.0
        )

        return {
            "tick": self.current_tick,
            "accepted_jobs": self.accepted_jobs,
            "completed_jobs": len(self.completed_jobs),
            "rejected_jobs": self.rejected_jobs,
            "sla_violations": self.sla_violation_count,
            "priority_weighted_completion_rate": completion_rate,
            "max_wait_by_tier": dict(self.max_wait_by_tier),
            "average_cpu_utilization": avg_cpu,
            "average_memory_utilization": avg_mem,
            "average_queue_length": float(len(self.queue)),
            "average_response_time": avg_response_time,
            "episode_reward": self.episode_reward,
            "reward_components_total": dict(self.reward_components_total),
            **self.last_step_info,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cloud_env.py -v`
Expected: `6 passed`

- [ ] **Step 5: Run the full test suite so far**

Run: `pytest -v`
Expected: all tests from Tasks 1–9 pass (no regressions).

- [ ] **Step 6: Commit**

```bash
git add environment/cloud_env.py tests/test_cloud_env.py
git commit -m "feat: rewrite CloudResourceEnv with 24-dim obs, global queue retry, v1.5 reward"
```

---

## Task 10: Scheduler updates — adapt existing baselines, add priority-aware Least Loaded

**Files:**
- Modify: `schedulers/least_loaded.py`
- Modify: `schedulers/random_scheduler.py`
- Modify: `schedulers/round_robin.py`
- Create: `schedulers/priority_least_loaded.py`
- Create: `tests/test_schedulers.py`

**Interfaces:**
- Consumes: `CloudResourceEnv` (Task 9).
- Produces: every scheduler class exposes `name: str`, `queue_retry_order: str`, `select_action(env) -> int`. `LeastLoadedScheduler.queue_retry_order = "fifo"` (priority-blind). `PriorityLeastLoadedScheduler.queue_retry_order = "priority_fifo"`. `RandomScheduler`/`RoundRobinScheduler.queue_retry_order = "priority_fifo"`.

**Design note (resolves an internal ambiguity in the source build prompt):** Section 5.3 of the build prompt says queue retry order is "the mechanism, not the agent" — i.e., environment-owned, not something a scheduler manually reorders. But Section 7 describes `priority_least_loaded.py` as differing from `least_loaded.py` specifically in "how it retries queued jobs" — while also saying both use identical logic for the incoming job ("same core logic... pick lowest utilization that fits"). Both are literally true if the *scheduler* only supplies a `queue_retry_order` value that the *environment's* fixed retry mechanism (built in Task 9) reads — the environment still does the sorting and placement (satisfying 5.3), while the scheduler choice still determines which retry order is active (satisfying Section 7's framing) and gives the priority-blind vs priority-aware baselines a genuine behavioral difference to contrast, per the acceptance criteria.

- [ ] **Step 1: Write the failing test**

Create `tests/test_schedulers.py`:

```python
"""Scheduler baselines: correct action indices, and blind vs. priority-aware retry order."""

from schedulers.least_loaded import LeastLoadedScheduler
from schedulers.priority_least_loaded import PriorityLeastLoadedScheduler
from schedulers.random_scheduler import RandomScheduler
from schedulers.round_robin import RoundRobinScheduler
from environment.cloud_env import CloudResourceEnv
from utils.config_v15 import load_config_v15


def test_least_loaded_picks_index_of_least_utilized_server():
    env = CloudResourceEnv(load_config_v15())
    env.reset(seed=1)
    env.servers[2].used = {"cpu": 0.0, "mem": 0.0}
    env.servers[0].used = {"cpu": 15.0, "mem": 60.0}

    action = LeastLoadedScheduler().select_action(env)

    assert action == 2


def test_least_loaded_is_priority_blind():
    assert LeastLoadedScheduler.queue_retry_order == "fifo"


def test_priority_least_loaded_is_priority_aware():
    assert PriorityLeastLoadedScheduler.queue_retry_order == "priority_fifo"


def test_priority_least_loaded_same_selection_as_least_loaded():
    env = CloudResourceEnv(load_config_v15())
    env.reset(seed=1)
    env.servers[2].used = {"cpu": 0.0, "mem": 0.0}
    env.servers[0].used = {"cpu": 15.0, "mem": 60.0}

    assert PriorityLeastLoadedScheduler().select_action(env) == LeastLoadedScheduler().select_action(env)


def test_random_scheduler_returns_valid_index():
    env = CloudResourceEnv(load_config_v15())
    env.reset(seed=1)
    scheduler = RandomScheduler(seed=1)

    for _ in range(20):
        action = scheduler.select_action(env)
        assert 0 <= action < len(env.servers)


def test_round_robin_cycles_through_all_servers():
    env = CloudResourceEnv(load_config_v15())
    env.reset(seed=1)
    scheduler = RoundRobinScheduler()

    actions = [scheduler.select_action(env) for _ in range(len(env.servers))]
    assert actions == [0, 1, 2, 3]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_schedulers.py -v`
Expected: `AttributeError: 'Server' object has no attribute 'load_score'` (or `ModuleNotFoundError` for `priority_least_loaded`)

- [ ] **Step 3: Update schedulers/least_loaded.py**

```python
"""Least-loaded baseline scheduler (priority-blind)."""

from __future__ import annotations

from environment.cloud_env import CloudResourceEnv


class LeastLoadedScheduler:
    """Select the least-utilized server; retries queued jobs in plain FIFO order, ignoring tier."""

    name = "Least Loaded"
    queue_retry_order = "fifo"

    def select_action(self, env: CloudResourceEnv) -> int:
        """Return the index of the least-loaded server."""
        best = min(env.servers, key=lambda server: server.average_utilization)
        return env.servers.index(best)
```

- [ ] **Step 4: Update schedulers/random_scheduler.py**

```python
"""Random baseline scheduler."""

from __future__ import annotations

import numpy as np

from environment.cloud_env import CloudResourceEnv


class RandomScheduler:
    """Select a server uniformly at random."""

    name = "Random"
    queue_retry_order = "priority_fifo"

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def select_action(self, env: CloudResourceEnv) -> int:
        """Return a random valid server index."""
        return int(self.rng.integers(0, len(env.servers)))
```

- [ ] **Step 5: Update schedulers/round_robin.py**

```python
"""Round-robin baseline scheduler."""

from __future__ import annotations

from environment.cloud_env import CloudResourceEnv


class RoundRobinScheduler:
    """Cycle through servers in order."""

    name = "Round Robin"
    queue_retry_order = "priority_fifo"

    def __init__(self) -> None:
        self.next_server = 0

    def reset(self) -> None:
        """Reset the next selected server to the first server."""
        self.next_server = 0

    def select_action(self, env: CloudResourceEnv) -> int:
        """Return the next server index in cyclic order."""
        action = self.next_server
        self.next_server = (self.next_server + 1) % len(env.servers)
        return action
```

- [ ] **Step 6: Create schedulers/priority_least_loaded.py**

```python
"""Priority-aware least-loaded baseline scheduler."""

from __future__ import annotations

from environment.cloud_env import CloudResourceEnv


class PriorityLeastLoadedScheduler:
    """Select the least-utilized server for the incoming job; queue retries are priority-then-FIFO."""

    name = "Priority-Aware Least Loaded"
    queue_retry_order = "priority_fifo"

    def select_action(self, env: CloudResourceEnv) -> int:
        """Return the index of the least-loaded server (identical rule to LeastLoadedScheduler)."""
        best = min(env.servers, key=lambda server: server.average_utilization)
        return env.servers.index(best)
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_schedulers.py -v`
Expected: `6 passed`

- [ ] **Step 8: Commit**

```bash
git add schedulers/least_loaded.py schedulers/random_scheduler.py schedulers/round_robin.py schedulers/priority_least_loaded.py tests/test_schedulers.py
git commit -m "feat: adapt baseline schedulers to dict-based Server, add priority-aware Least Loaded"
```

---

## Task 11: Integration test — one full episode per scheduler, no exceptions

**Files:**
- Create: `tests/test_integration.py`

**Interfaces:**
- Consumes: `CloudResourceEnv` (Task 9), all scheduler classes (Task 10), `stable_baselines3.PPO`, `rl.model.build_ppo_model`.

- [ ] **Step 1: Write the integration test**

Create `tests/test_integration.py`:

```python
"""Integration: one full headless episode per scheduler, including untrained PPO, no exceptions."""

from stable_baselines3 import PPO

from environment.cloud_env import CloudResourceEnv
from schedulers.least_loaded import LeastLoadedScheduler
from schedulers.priority_least_loaded import PriorityLeastLoadedScheduler
from schedulers.random_scheduler import RandomScheduler
from schedulers.round_robin import RoundRobinScheduler
from utils.config_v15 import load_config_v15


def _run_episode(scheduler) -> None:
    config = load_config_v15()
    config.episode_length = 100
    env = CloudResourceEnv(config, queue_retry_order=scheduler.queue_retry_order)
    env.reset(seed=1)
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = scheduler.select_action(env)
        _, _, terminated, truncated, info = env.step(action)

    assert info["tick"] == config.episode_length


def test_random_scheduler_full_episode():
    _run_episode(RandomScheduler(seed=1))


def test_round_robin_full_episode():
    _run_episode(RoundRobinScheduler())


def test_least_loaded_full_episode():
    _run_episode(LeastLoadedScheduler())


def test_priority_least_loaded_full_episode():
    _run_episode(PriorityLeastLoadedScheduler())


def test_untrained_ppo_full_episode():
    config = load_config_v15()
    config.episode_length = 100
    env = CloudResourceEnv(config)
    model = PPO("MlpPolicy", env, verbose=0, seed=1)

    observation, _ = env.reset(seed=1)
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = env.step(int(action))

    assert info["tick"] == config.episode_length
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_integration.py -v`
Expected: `5 passed` (PPO's own construction is enough to prove obs/action-space compatibility — no crash across a full 100-tick episode for any scheduler)

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add full-episode integration test for every scheduler including untrained PPO"
```

---

## Task 12: `metrics/metrics.py` — v1.5 aggregator additions

**Files:**
- Modify: `metrics/metrics.py` (append; do not touch the existing v1 `EPISODE_METRIC_COLUMNS`/`summarize_episode`/`aggregate_metrics`)
- Create: `tests/test_metrics_v15.py`

**Interfaces:**
- Consumes: `info` dict shape produced by `CloudResourceEnv._get_info()` (Task 9).
- Produces: `EPISODE_METRIC_COLUMNS_V15: list[str]`, `summarize_episode_v15(scheduler, episode, info) -> dict`, `aggregate_metrics_v15(rows) -> pd.DataFrame`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_metrics_v15.py`:

```python
"""v1.5 metrics: SLA rate, priority-weighted completion, max-wait-per-tier, reward components."""

from metrics.metrics import aggregate_metrics_v15, summarize_episode_v15


def _fake_info() -> dict:
    return {
        "tick": 500,
        "accepted_jobs": 400,
        "completed_jobs": 380,
        "rejected_jobs": 20,
        "sla_violations": 10,
        "priority_weighted_completion_rate": 0.87,
        "max_wait_by_tier": {"gold": 4, "silver": 8, "bronze": 22},
        "average_cpu_utilization": 0.6,
        "average_memory_utilization": 0.55,
        "average_queue_length": 3.2,
        "average_response_time": 12.5,
        "episode_reward": 245.0,
        "reward_components_total": {
            "completion": 300.0,
            "sla_violation": -20.0,
            "rejection": -20.0,
            "overload": -5.0,
            "queue": -1.5,
            "balance": 10.0,
            "starvation_bonus": 2.5,
        },
    }


def test_summarize_episode_v15_computes_sla_rate():
    row = summarize_episode_v15("PPO Agent", 0, _fake_info())

    total_jobs = 400 + 20 + 380
    assert row["sla_violation_rate"] == 10 / total_jobs
    assert row["max_wait_bronze"] == 22
    assert row["reward_completion"] == 300.0


def test_aggregate_metrics_v15_sorts_by_sla_violation_rate():
    rows = [
        summarize_episode_v15("A", 0, _fake_info()),
        summarize_episode_v15("B", 0, {**_fake_info(), "sla_violations": 1}),
    ]

    summary = aggregate_metrics_v15(rows)

    assert list(summary["scheduler"])[0] == "B"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_metrics_v15.py -v`
Expected: `ImportError: cannot import name 'summarize_episode_v15'`

- [ ] **Step 3: Append to metrics/metrics.py**

Append to `metrics/metrics.py` (keep the existing v1 constants/functions untouched above this):

```python
EPISODE_METRIC_COLUMNS_V15 = [
    "scheduler",
    "episode",
    "average_response_time",
    "average_queue_length",
    "successful_jobs",
    "rejected_jobs",
    "sla_violations",
    "sla_violation_rate",
    "priority_weighted_completion_rate",
    "max_wait_gold",
    "max_wait_silver",
    "max_wait_bronze",
    "average_cpu_utilization",
    "average_memory_utilization",
    "episode_reward",
    "reward_completion",
    "reward_sla_violation",
    "reward_rejection",
    "reward_overload",
    "reward_queue",
    "reward_balance",
    "reward_starvation_bonus",
]


def summarize_episode_v15(
    scheduler: str,
    episode: int,
    info: dict,
) -> dict[str, float | int | str]:
    """Convert final v1.5 environment info into one episode metrics row."""
    total_jobs = info["accepted_jobs"] + info["rejected_jobs"] + info["completed_jobs"]
    sla_rate = info["sla_violations"] / total_jobs if total_jobs else 0.0
    reward_totals = info["reward_components_total"]
    return {
        "scheduler": scheduler,
        "episode": episode,
        "average_response_time": float(info["average_response_time"]),
        "average_queue_length": float(info["average_queue_length"]),
        "successful_jobs": int(info["completed_jobs"]),
        "rejected_jobs": int(info["rejected_jobs"]),
        "sla_violations": int(info["sla_violations"]),
        "sla_violation_rate": float(sla_rate),
        "priority_weighted_completion_rate": float(info["priority_weighted_completion_rate"]),
        "max_wait_gold": int(info["max_wait_by_tier"]["gold"]),
        "max_wait_silver": int(info["max_wait_by_tier"]["silver"]),
        "max_wait_bronze": int(info["max_wait_by_tier"]["bronze"]),
        "average_cpu_utilization": float(info["average_cpu_utilization"]),
        "average_memory_utilization": float(info["average_memory_utilization"]),
        "episode_reward": float(info["episode_reward"]),
        "reward_completion": float(reward_totals.get("completion", 0.0)),
        "reward_sla_violation": float(reward_totals.get("sla_violation", 0.0)),
        "reward_rejection": float(reward_totals.get("rejection", 0.0)),
        "reward_overload": float(reward_totals.get("overload", 0.0)),
        "reward_queue": float(reward_totals.get("queue", 0.0)),
        "reward_balance": float(reward_totals.get("balance", 0.0)),
        "reward_starvation_bonus": float(reward_totals.get("starvation_bonus", 0.0)),
    }


def aggregate_metrics_v15(rows: Iterable[dict[str, float | int | str]]) -> pd.DataFrame:
    """Aggregate v1.5 per-episode rows by scheduler, sorted by SLA violation rate ascending."""
    frame = pd.DataFrame(rows, columns=EPISODE_METRIC_COLUMNS_V15)
    agg_columns = {
        col: (col, "mean") for col in EPISODE_METRIC_COLUMNS_V15 if col not in ("scheduler", "episode")
    }
    return frame.groupby("scheduler", as_index=False).agg(**agg_columns).sort_values("sla_violation_rate")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_metrics_v15.py -v`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add metrics/metrics.py tests/test_metrics_v15.py
git commit -m "feat: add v1.5 metrics aggregator (SLA rate, priority-weighted completion, max-wait)"
```

---

## Task 13: `rl/train.py` v1.5 — retrain entry point with tier-aware logging

**Files:**
- Modify: `rl/train.py`

**Interfaces:**
- Consumes: `AppConfigV15` (Task 3), `CloudResourceEnv` (Task 9), `build_ppo_model` (Task 1).
- Produces: `train_agent_v15(config: AppConfigV15, console: Console) -> Path` (returns the saved `.zip` path). Keeps the existing v1 `train_agent`/`RichTrainingCallback` untouched above this addition; the Task 1 `RewardComponentCallback` is reused as-is (its `reward_components` key is unchanged between v1 and v1.5).

- [ ] **Step 1: Append train_agent_v15 to rl/train.py**

Add this import near the top:

```python
from utils.config_v15 import AppConfigV15
```

Append to the bottom of `rl/train.py`:

```python
def train_agent_v15(config: AppConfigV15, console: Console) -> Path:
    """Train PPO on the v1.5 heterogeneous, priority-aware environment."""
    from environment.cloud_env import CloudResourceEnv as CloudResourceEnvV15

    models_dir = Path(config.models_dir)
    results_dir = Path(config.results_dir)
    log_dir = results_dir / "training_logs"
    tensorboard_dir = results_dir / "tensorboard"
    best_dir = models_dir / "best"
    log_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    train_env = Monitor(CloudResourceEnvV15(config), filename=str(log_dir / "monitor.csv"))
    eval_env = Monitor(CloudResourceEnvV15(config))
    model = build_ppo_model(train_env, config.seed, tensorboard_log=tensorboard_dir)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(log_dir),
        eval_freq=max(config.episode_length, 1),
        deterministic=True,
        render=False,
    )
    callbacks = CallbackList(
        [RichTrainingCallback(console), RewardComponentCallback(), eval_callback]
    )

    console.print(
        f"[bold]Training PPO (v1.5)[/bold] for {config.training_timesteps:,} timesteps"
    )
    model.learn(total_timesteps=config.training_timesteps, callback=callbacks)

    final_model_path = models_dir / "ppo_v15_final"
    model.save(final_model_path)
    plot_training_reward(log_dir, results_dir)
    console.print(f"Saved final model to [green]{final_model_path}.zip[/green]")
    return final_model_path.with_suffix(".zip")
```

Note: `CloudResourceEnv` is imported both at module level (from Task 1's `environment.cloud_env`, still v1-shaped at that point in history) and locally here as `CloudResourceEnvV15` — by the time this task runs, Task 9 has already replaced `environment/cloud_env.py` in place, so the module-level import in the existing v1 `train_agent` function now also points at the v1.5 class. Since `train_agent`'s signature takes `AppConfig` (v1) but the class now expects `AppConfigV15`, **v1's `train_agent`/`AppConfig` path is no longer runnable after Task 9** — this is expected (Global Constraints: config.yaml stays only for reproducibility record, not as a live code path; see spec Section 2 reconciliation). Do not attempt to keep `train_agent` (v1) working; only `train_agent_v15` is exercised going forward.

- [ ] **Step 2: Smoke-test the new entry point with a tiny run**

```bash
python -c "from rich.console import Console; from utils.config_v15 import load_config_v15, ensure_directories_v15; from rl.train import train_agent_v15; c=load_config_v15(); c.training_timesteps=2000; ensure_directories_v15(c); train_agent_v15(c, Console())"
```

Expected: runs to completion with no exceptions, prints per-episode Rich rows including `sla_violations` and `max_wait G/S/B`, saves `models/v1.5/ppo_v15_final.zip`.

- [ ] **Step 3: Commit**

```bash
git add rl/train.py
git commit -m "feat: add train_agent_v15 entry point for the heterogeneous environment"
```

---

## Task 14: `rl/evaluate.py` v1.5 — evaluate all 5 schedulers

**Files:**
- Modify: `rl/evaluate.py`

**Interfaces:**
- Consumes: `CloudResourceEnv` (Task 9), all scheduler classes (Task 10), `aggregate_metrics_v15`/`summarize_episode_v15` (Task 12), `AppConfigV15` (Task 3).
- Produces: `evaluate_all_v15(config: AppConfigV15, model_path, console: Console) -> pd.DataFrame`.

- [ ] **Step 1: Append v1.5 evaluation functions to rl/evaluate.py**

Add these imports near the top:

```python
from environment.cloud_env import CloudResourceEnv as CloudResourceEnvV15
from metrics.metrics import aggregate_metrics_v15, summarize_episode_v15
from schedulers.priority_least_loaded import PriorityLeastLoadedScheduler
from utils.config_v15 import AppConfigV15
```

Append to the bottom of `rl/evaluate.py`:

```python
def evaluate_all_v15(config: AppConfigV15, model_path: str | Path, console: Console) -> pd.DataFrame:
    """Evaluate all v1.5 baselines and PPO, save CSVs and plots, and return a summary."""
    rows: list[dict[str, float | int | str]] = []
    schedulers = [
        RandomScheduler(config.seed),
        RoundRobinScheduler(),
        LeastLoadedScheduler(),
        PriorityLeastLoadedScheduler(),
    ]

    for scheduler in schedulers:
        rows.extend(_evaluate_scheduler_v15(config, scheduler))

    ppo_model = PPO.load(model_path)
    rows.extend(_evaluate_ppo_v15(config, ppo_model))

    episodes = pd.DataFrame(rows)
    summary = aggregate_metrics_v15(rows)
    results_dir = Path(config.results_dir)
    episodes.to_csv(results_dir / "evaluation_episodes.csv", index=False)
    summary.to_csv(results_dir / "evaluation_summary.csv", index=False)
    plot_evaluation_metrics(summary, results_dir)
    _print_summary_v15(summary, console)
    return summary


def _evaluate_scheduler_v15(config: AppConfigV15, scheduler) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for episode in range(config.evaluation_episodes):
        if hasattr(scheduler, "reset"):
            scheduler.reset()

        env = CloudResourceEnvV15(config, queue_retry_order=scheduler.queue_retry_order)
        _, info = env.reset(seed=config.seed + episode)
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = scheduler.select_action(env)
            _, _, terminated, truncated, info = env.step(action)

        rows.append(summarize_episode_v15(scheduler.name, episode, info))
    return rows


def _evaluate_ppo_v15(config: AppConfigV15, model: PPO) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for episode in range(config.evaluation_episodes):
        env = CloudResourceEnvV15(config, queue_retry_order="priority_fifo")
        observation, info = env.reset(seed=config.seed + episode)
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(int(action))

        rows.append(summarize_episode_v15("PPO Agent", episode, info))
    return rows


def _print_summary_v15(summary: pd.DataFrame, console: Console) -> None:
    table = Table(title="CloudPilot v1.5 Evaluation Summary")
    for column in summary.columns:
        table.add_column(column.replace("_", " ").title())

    for _, row in summary.iterrows():
        table.add_row(
            *(f"{value:.3f}" if isinstance(value, float) else str(value) for value in row)
        )
    console.print(table)
```

Note: same caveat as Task 13 — the module-level `CloudResourceEnv` import at the top of `rl/evaluate.py` (used by the existing v1 `evaluate_all`/`_evaluate_scheduler`/`_evaluate_ppo`) now resolves to the v1.5 class after Task 9, so v1's `evaluate_all` is no longer runnable. Only `evaluate_all_v15` is exercised going forward.

- [ ] **Step 2: Verify the module imports cleanly**

```bash
python -c "from rl.evaluate import evaluate_all_v15; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add rl/evaluate.py
git commit -m "feat: add evaluate_all_v15 covering all five schedulers"
```

---

## Task 15: `.gitignore` — allow `results/v1.5/` to be committed

**Files:**
- Modify: `.gitignore`

**Interfaces:** none (build config only).

- [ ] **Step 1: Add an exception for results/v1.5/**

Current `.gitignore` has a blanket `results/` entry, which would silently prevent `results/v1.5/*.csv` from ever being committed — required by the acceptance criteria. Replace the single line:

```
results/
```

with:

```
results/
!results/v1.5/
!results/v1.5/**
results/v1.5/tensorboard/
results/v1.5/training_logs/
```

(Keeps `results/` ignored by default, un-ignores everything under `results/v1.5/` for the committed CSVs/plots, then re-ignores the large TensorBoard/monitor logs specifically so only the small CSV/PNG artifacts get committed.)

- [ ] **Step 2: Verify with git status**

```bash
mkdir -p results/v1.5 && touch results/v1.5/test.csv && git status --porcelain results/
```

Expected: `results/v1.5/test.csv` shows as untracked (visible), not silently ignored. Then remove the test file:

```bash
rm results/v1.5/test.csv
```

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: un-ignore results/v1.5/ so evaluation CSVs can be committed"
```

---

## Task 16: Backend API — dict-based schemas, v1.5 config, new WebSocket fields

**Files:**
- Modify: `backend/api/schemas.py`
- Modify: `backend/api/simulation.py`
- Modify: `backend/api/app.py`

**Interfaces:**
- Consumes: `CloudResourceEnv` (Task 9), all schedulers (Task 10), `AppConfigV15`/`load_config_v15` (Task 3).
- Produces: `JobState` gains `id: str` (was `job_id: int`), `priority_tier: str`, `deadline_ticks: int`, `ticks_waited: int`; drops `cpu_required`/`memory_required` naming ambiguity (kept as-is for frontend continuity, now sourced from `job.requirements`). `ServerState` drops `queue_length` (queue is global now). `MetricsState` gains `sla_violations_total: int`, `priority_weighted_completion_rate: float`, `max_wait_by_tier: dict[str, int]`. `ChartPoint` gains `sla_violations_total: int`.

- [ ] **Step 1: Rewrite backend/api/schemas.py**

```python
"""Pydantic schemas returned by the CloudPilot API."""

from pydantic import BaseModel


class JobState(BaseModel):
    """Current incoming or running job state."""

    id: str
    cpu_required: float
    memory_required: float
    runtime: int
    arrival_tick: int
    priority_tier: str
    deadline_ticks: int
    ticks_waited: int
    remaining_time: int | None = None


class ServerState(BaseModel):
    """Serializable server state for the dashboard."""

    id: str
    name: str
    cpu_utilization: float
    memory_utilization: float
    running_jobs: list[JobState]
    status: str


class MetricsState(BaseModel):
    """Live KPI snapshot."""

    jobs_processed: int
    jobs_completed: int
    jobs_rejected: int
    sla_violations_total: int
    priority_weighted_completion_rate: float
    max_wait_by_tier: dict[str, int]
    average_response_time: float
    average_queue_length: float
    average_cpu_utilization: float
    average_memory_utilization: float
    current_reward: float
    episode_reward: float


class DecisionState(BaseModel):
    """Most recent scheduling decision."""

    job_id: str | None = None
    scheduler: str
    assigned_server: int | None = None
    accepted: bool = False
    reward: float = 0.0


class ChartPoint(BaseModel):
    """One time-series point for live charts."""

    step: int
    episode: int
    reward: float
    average_queue_length: float
    jobs_completed: int
    jobs_rejected: int
    sla_violations_total: int
    cpu: list[float]
    memory: list[float]


class SimulationState(BaseModel):
    """Complete state streamed to dashboard clients."""

    project_name: str = "CloudPilot"
    status: str
    scheduler: str
    episode: int
    step: int
    speed: float
    incoming_job: JobState | None
    servers: list[ServerState]
    decision: DecisionState
    metrics: MetricsState
    history: list[ChartPoint]
    heatmap: list[list[float]]


class ControlRequest(BaseModel):
    """Simulation control request."""

    scheduler: str | None = None
    speed: float | None = None


class TrainingState(BaseModel):
    """Training progress fields displayed by the dashboard."""

    current_episode: int = 0
    current_timestep: int = 0
    average_reward: float = 0.0
    best_reward: float = 0.0
    loss: float | None = None
    status: str = "idle"
```

- [ ] **Step 2: Rewrite backend/api/simulation.py**

```python
"""Live simulation runner used by REST and WebSocket endpoints."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

from backend.api.schemas import (
    ChartPoint,
    DecisionState,
    JobState,
    MetricsState,
    ServerState,
    SimulationState,
    TrainingState,
)
from environment.cloud_env import CloudResourceEnv
from schedulers.least_loaded import LeastLoadedScheduler
from schedulers.priority_least_loaded import PriorityLeastLoadedScheduler
from schedulers.random_scheduler import RandomScheduler
from schedulers.round_robin import RoundRobinScheduler
from utils.config_v15 import AppConfigV15

if TYPE_CHECKING:
    from stable_baselines3 import PPO


class LiveSimulation:
    """Manage one live CloudPilot simulation loop."""

    def __init__(self, config: AppConfigV15) -> None:
        self.config = config
        self.env = CloudResourceEnv(config)
        self.observation, self.info = self.env.reset(seed=config.seed)
        self.status = "Paused"
        self.scheduler_name = "Priority-Aware Least Loaded"
        self.speed = config.simulation_speed
        self.episode = 1
        self.history: list[ChartPoint] = []
        self.heatmap: list[list[float]] = [[] for _ in self.env.servers]
        self.decision = DecisionState(scheduler=self.scheduler_name)
        self.training = TrainingState()
        self._task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._round_robin = RoundRobinScheduler()
        self._random = RandomScheduler(config.seed)
        self._least_loaded = LeastLoadedScheduler()
        self._priority_least_loaded = PriorityLeastLoadedScheduler()
        self._ppo_model: PPO | None = self._load_ppo_model()

    async def start(self) -> None:
        """Start the background simulation loop."""
        self.status = "Running"
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def pause(self) -> None:
        """Pause simulation ticks."""
        self.status = "Paused"

    async def reset(self) -> None:
        """Reset simulation state."""
        async with self._lock:
            self.env = CloudResourceEnv(self.config, queue_retry_order=self._active_retry_order())
            self.observation, self.info = self.env.reset(seed=self.config.seed)
            self.episode = 1
            self.history.clear()
            self.heatmap = [[] for _ in self.env.servers]
            self.decision = DecisionState(scheduler=self.scheduler_name)
            self._round_robin.reset()

    async def configure(self, scheduler: str | None = None, speed: float | None = None) -> None:
        """Update active scheduler or tick speed."""
        async with self._lock:
            if scheduler is not None:
                self.scheduler_name = scheduler
                if scheduler == "Round Robin":
                    self._round_robin.reset()
                self.env.queue_retry_order = self._active_retry_order()
            if speed is not None:
                self.speed = max(0.25, min(speed, 20.0))
            self.decision.scheduler = self.scheduler_name

    async def snapshot(self) -> SimulationState:
        """Return the current dashboard state."""
        async with self._lock:
            return self._snapshot_unlocked()

    async def _run(self) -> None:
        while True:
            if self.status == "Running":
                async with self._lock:
                    self._tick_unlocked()
            await asyncio.sleep(1.0 / self.speed)

    def _tick_unlocked(self) -> None:
        incoming = self.env.current_job
        action = self._select_action()
        self.observation, reward, terminated, truncated, self.info = self.env.step(action)
        self.decision = DecisionState(
            job_id=incoming.id if incoming else None,
            scheduler=self.scheduler_name,
            assigned_server=action,
            accepted=bool(self.info.get("scheduled", False)),
            reward=float(reward),
        )
        self._record_history(reward)

        if terminated or truncated:
            self.episode += 1
            self.observation, self.info = self.env.reset(seed=self.config.seed + self.episode)
            self._round_robin.reset()

    def _active_retry_order(self) -> str:
        return "fifo" if self.scheduler_name == "Least Loaded" else "priority_fifo"

    def _select_action(self) -> int:
        if self.scheduler_name == "Random":
            return self._random.select_action(self.env)
        if self.scheduler_name == "Round Robin":
            return self._round_robin.select_action(self.env)
        if self.scheduler_name == "Least Loaded":
            return self._least_loaded.select_action(self.env)
        if self.scheduler_name == "PPO":
            if self._ppo_model is None:
                return self._priority_least_loaded.select_action(self.env)
            action, _ = self._ppo_model.predict(self.observation, deterministic=True)
            return int(action)
        return self._priority_least_loaded.select_action(self.env)

    def _record_history(self, reward: float) -> None:
        cpu = [server.utilization("cpu") for server in self.env.servers]
        memory = [server.utilization("mem") for server in self.env.servers]
        for index, value in enumerate(cpu):
            self.heatmap[index].append(value)
            self.heatmap[index] = self.heatmap[index][-60:]

        point = ChartPoint(
            step=self.env.current_tick,
            episode=self.episode,
            reward=float(reward),
            average_queue_length=float(self.info["average_queue_length"]),
            jobs_completed=int(self.info["completed_jobs"]),
            jobs_rejected=int(self.info["rejected_jobs"]),
            sla_violations_total=int(self.info["sla_violations"]),
            cpu=cpu,
            memory=memory,
        )
        self.history.append(point)
        self.history = self.history[-80:]

    def _snapshot_unlocked(self) -> SimulationState:
        return SimulationState(
            status=self.status,
            scheduler=self.scheduler_name,
            episode=self.episode,
            step=self.env.current_tick,
            speed=self.speed,
            incoming_job=self._job_state(self.env.current_job),
            servers=[self._server_state(server) for server in self.env.servers],
            decision=self.decision,
            metrics=self._metrics_state(),
            history=self.history,
            heatmap=self.heatmap,
        )

    def _server_state(self, server: Any) -> ServerState:
        cpu_util = server.utilization("cpu")
        mem_util = server.utilization("mem")
        if cpu_util >= 0.9 or mem_util >= 0.9:
            status = "Hot"
        elif server.running_jobs:
            status = "Active"
        else:
            status = "Idle"

        return ServerState(
            id=server.id,
            name=f"Server {server.id}",
            cpu_utilization=cpu_util,
            memory_utilization=mem_util,
            running_jobs=[self._job_state(job) for job in server.running_jobs],
            status=status,
        )

    def _metrics_state(self) -> MetricsState:
        return MetricsState(
            jobs_processed=int(self.info["accepted_jobs"] + self.info["rejected_jobs"]),
            jobs_completed=int(self.info["completed_jobs"]),
            jobs_rejected=int(self.info["rejected_jobs"]),
            sla_violations_total=int(self.info["sla_violations"]),
            priority_weighted_completion_rate=float(self.info["priority_weighted_completion_rate"]),
            max_wait_by_tier=dict(self.info["max_wait_by_tier"]),
            average_response_time=float(self.info["average_response_time"]),
            average_queue_length=float(self.info["average_queue_length"]),
            average_cpu_utilization=float(self.info["average_cpu_utilization"]),
            average_memory_utilization=float(self.info["average_memory_utilization"]),
            current_reward=float(self.info.get("reward", 0.0)),
            episode_reward=float(self.info["episode_reward"]),
        )

    @staticmethod
    def _job_state(job: Any | None) -> JobState | None:
        if job is None:
            return None
        return JobState(
            id=job.id,
            cpu_required=job.requirements["cpu"],
            memory_required=job.requirements["mem"],
            runtime=job.runtime,
            arrival_tick=job.arrival_tick,
            priority_tier=job.priority_tier.value,
            deadline_ticks=job.deadline_ticks,
            ticks_waited=job.ticks_waited,
            remaining_time=getattr(job, "remaining_time", None),
        )

    @staticmethod
    def _load_ppo_model() -> "PPO | None":
        model_path = Path("models/v1.5/ppo_v15_final.zip")
        if not model_path.exists():
            return None
        from stable_baselines3 import PPO

        return PPO.load(model_path)
```

- [ ] **Step 3: Update backend/api/app.py's config loader**

Change:

```python
from utils.helpers import load_config

config = load_config()
```

to:

```python
from utils.config_v15 import load_config_v15

config = load_config_v15()
```

- [ ] **Step 4: Verify the backend boots**

```bash
python -m backend.main &
sleep 2
curl http://127.0.0.1:8000/api/state
kill %1
```

Expected: JSON response with `servers` list of 4 entries (ids `A`/`B`/`C`/`D`), `incoming_job.priority_tier` present, no server errors in the console.

- [ ] **Step 5: Commit**

```bash
git add backend/api/schemas.py backend/api/simulation.py backend/api/app.py
git commit -m "feat: wire backend API to v1.5 config, dict-based schema, and priority metrics"
```

---

## Task 17: Frontend types + tailwind tier colors

**Files:**
- Modify: `frontend/src/types.ts`
- Modify: `frontend/tailwind.config.js`

**Interfaces:**
- Produces: `JobState` TS interface matches Task 16's schema (`id`, `priority_tier`, `deadline_ticks`, `ticks_waited`). `ServerState` drops `queue_length`. `MetricsState` gains `sla_violations_total`, `priority_weighted_completion_rate`, `max_wait_by_tier`. `ChartPoint` gains `sla_violations_total`. Tailwind gains `gold`, `silver`, `bronze` colors.

- [ ] **Step 1: Rewrite frontend/src/types.ts**

```typescript
export interface JobState {
  id: string;
  cpu_required: number;
  memory_required: number;
  runtime: number;
  arrival_tick: number;
  priority_tier: string;
  deadline_ticks: number;
  ticks_waited: number;
  remaining_time: number | null;
}

export interface ServerState {
  id: string;
  name: string;
  cpu_utilization: number;
  memory_utilization: number;
  running_jobs: JobState[];
  status: string;
}

export interface MetricsState {
  jobs_processed: number;
  jobs_completed: number;
  jobs_rejected: number;
  sla_violations_total: number;
  priority_weighted_completion_rate: number;
  max_wait_by_tier: Record<string, number>;
  average_response_time: number;
  average_queue_length: number;
  average_cpu_utilization: number;
  average_memory_utilization: number;
  current_reward: number;
  episode_reward: number;
}

export interface DecisionState {
  job_id: string | null;
  scheduler: string;
  assigned_server: number | null;
  accepted: boolean;
  reward: number;
}

export interface ChartPoint {
  step: number;
  episode: number;
  reward: number;
  average_queue_length: number;
  jobs_completed: number;
  jobs_rejected: number;
  sla_violations_total: number;
  cpu: number[];
  memory: number[];
}

export interface SimulationState {
  project_name: string;
  status: string;
  scheduler: string;
  episode: number;
  step: number;
  speed: number;
  incoming_job: JobState | null;
  servers: ServerState[];
  decision: DecisionState;
  metrics: MetricsState;
  history: ChartPoint[];
  heatmap: number[][];
}

export interface TrainingState {
  current_episode: number;
  current_timestep: number;
  average_reward: number;
  best_reward: number;
  loss: number | null;
  status: string;
}
```

- [ ] **Step 2: Add tier colors to frontend/tailwind.config.js**

```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"]
      },
      colors: {
        ink: "#17202a",
        cloud: "#f5f7fb",
        panel: "#ffffff",
        line: "#d8dee9",
        teal: "#0f9f9a",
        cobalt: "#2d6cdf",
        amber: "#d99025",
        rose: "#d1495b",
        gold: "#c9971c",
        silver: "#8a94a6",
        bronze: "#a1622f"
      },
      boxShadow: {
        soft: "0 10px 30px rgba(23, 32, 42, 0.08)"
      }
    }
  },
  plugins: []
};
```

- [ ] **Step 3: Verify the frontend still type-checks**

```bash
cd frontend && npx tsc --noEmit
```

Expected: errors listing every file that still references the old field names (`job_id`, `cpu_required` is fine/unchanged, `queue_length` on `ServerState`) — this is expected and resolved in Task 18. Note the error list before moving on.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/types.ts frontend/tailwind.config.js
git commit -m "feat: update frontend types for dict-based schema, add tier colors"
```

---

## Task 18: Frontend components — tier color-coding, new panels, scheduler option

**Files:**
- Modify: `frontend/src/components/ServerGrid.tsx`
- Modify: `frontend/src/components/IncomingJobPanel.tsx`
- Modify: `frontend/src/components/JobFlow.tsx`
- Modify: `frontend/src/components/Header.tsx`
- Modify: `frontend/src/components/MetricsPanel.tsx`
- Create: `frontend/src/components/TierWaitPanel.tsx`
- Modify: `frontend/src/pages/App.tsx`

**Interfaces:**
- Consumes: types from Task 17.
- Produces: `TierWaitPanel({ maxWaitByTier: Record<string, number> })` component.

- [ ] **Step 1: Update ServerGrid.tsx — drop the Queue tile, color job chips by tier**

```typescript
import { Server } from "lucide-react";

import type { ServerState } from "../types";
import { UtilizationBar } from "./UtilizationBar";

interface ServerGridProps {
  servers: ServerState[];
  selectedServer: number | null;
}

const tierClass: Record<string, string> = {
  gold: "border-gold/50 bg-gold/10 text-gold",
  silver: "border-silver/50 bg-silver/10 text-silver",
  bronze: "border-bronze/50 bg-bronze/10 text-bronze"
};

export function ServerGrid({ servers, selectedServer }: ServerGridProps) {
  return (
    <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      {servers.map((server, index) => (
        <article
          key={server.id}
          className={`panel transition duration-300 ${
            selectedServer === index ? "ring-2 ring-teal" : ""
          }`}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Server size={18} className="text-cobalt" />
              <h2 className="text-base font-semibold text-ink">{server.name}</h2>
            </div>
            <span className={statusClass(server.status)}>{server.status}</span>
          </div>

          <div className="mt-5 space-y-4">
            <UtilizationBar label="CPU" value={server.cpu_utilization} tone="cpu" />
            <UtilizationBar label="Memory" value={server.memory_utilization} tone="memory" />
          </div>

          <div className="mt-5 grid grid-cols-1 gap-3">
            <div className="metric-tile">
              <span>Running</span>
              <strong>{server.running_jobs.length}</strong>
            </div>
          </div>

          <div className="mt-4 flex min-h-9 flex-wrap gap-1">
            {server.running_jobs.slice(0, 8).map((job) => (
              <span
                key={job.id}
                className={`job-chip ${tierClass[job.priority_tier] ?? ""}`}
              >
                #{job.id}
              </span>
            ))}
          </div>
        </article>
      ))}
    </section>
  );
}

function statusClass(status: string) {
  if (status === "Hot") {
    return "rounded bg-rose/10 px-2 py-1 text-xs font-semibold text-rose";
  }
  if (status === "Active") {
    return "rounded bg-teal/10 px-2 py-1 text-xs font-semibold text-teal";
  }
  return "rounded bg-slate-100 px-2 py-1 text-xs font-semibold text-slate-500";
}
```

Note: `selectedServer` is compared against `index` now (array position), not `server.id` (a string letter) — matches `DecisionState.assigned_server`, which is the action integer index, not a server id.

- [ ] **Step 2: Update IncomingJobPanel.tsx — tier badge, deadline/ticks_waited**

```typescript
import { Cpu, MemoryStick, Timer } from "lucide-react";
import type React from "react";

import type { JobState } from "../types";

interface IncomingJobPanelProps {
  job: JobState | null;
}

const tierBadgeClass: Record<string, string> = {
  gold: "bg-gold/10 text-gold border-gold/40",
  silver: "bg-silver/10 text-silver border-silver/40",
  bronze: "bg-bronze/10 text-bronze border-bronze/40"
};

export function IncomingJobPanel({ job }: IncomingJobPanelProps) {
  return (
    <section className="panel min-h-[180px]">
      <div className="section-title">Incoming Job</div>
      {job ? (
        <div key={job.id} className="mt-5 animate-pop rounded border border-line bg-cloud p-4">
          <div className="flex items-center justify-between">
            <div className="text-xl font-semibold text-ink">Job #{job.id}</div>
            <span
              className={`rounded border px-2 py-1 text-xs font-semibold uppercase ${
                tierBadgeClass[job.priority_tier] ?? ""
              }`}
            >
              {job.priority_tier}
            </span>
          </div>
          <div className="mt-4 grid grid-cols-3 gap-3">
            <MetricIcon icon={<Cpu size={16} />} label="CPU" value={`${job.cpu_required.toFixed(1)}%`} />
            <MetricIcon icon={<MemoryStick size={16} />} label="Memory" value={`${job.memory_required.toFixed(1)} GB`} />
            <MetricIcon icon={<Timer size={16} />} label="Runtime" value={`${job.runtime}`} />
          </div>
          <div className="mt-3 flex justify-between text-xs text-slate-500">
            <span>Deadline: {job.deadline_ticks} ticks</span>
            <span>Waited: {job.ticks_waited} ticks</span>
          </div>
        </div>
      ) : (
        <div className="mt-8 text-sm text-slate-500">No active job</div>
      )}
    </section>
  );
}

function MetricIcon({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="rounded border border-line bg-panel p-3">
      <div className="flex items-center gap-2 text-xs text-slate-500">
        {icon}
        {label}
      </div>
      <div className="mt-1 text-sm font-semibold text-ink">{value}</div>
    </div>
  );
}
```

- [ ] **Step 3: Update JobFlow.tsx for the `id` rename**

```typescript
import type { DecisionState, JobState } from "../types";

interface JobFlowProps {
  job: JobState | null;
  decision: DecisionState;
}

export function JobFlow({ job, decision }: JobFlowProps) {
  const target =
    decision.assigned_server === null
      ? "Waiting"
      : `Server ${String.fromCharCode(65 + decision.assigned_server)}`;

  return (
    <section className="panel">
      <div className="section-title">Job Flow</div>
      <div className="flow-track mt-5">
        <span>Incoming</span>
        <span>CloudPilot</span>
        <span>{target}</span>
        {job && <div key={job.id} className="flow-dot" />}
      </div>
    </section>
  );
}
```

- [ ] **Step 4: Add the priority-aware scheduler option to Header.tsx**

Change:

```typescript
const schedulers = ["Random", "Round Robin", "Least Loaded", "PPO"];
```

to:

```typescript
const schedulers = ["Random", "Round Robin", "Least Loaded", "Priority-Aware Least Loaded", "PPO"];
```

- [ ] **Step 5: Add SLA/priority-weighted-completion tiles to MetricsPanel.tsx**

```typescript
import type { MetricsState } from "../types";

interface MetricsPanelProps {
  metrics: MetricsState;
}

const metricMap: Array<[keyof MetricsState, string, (value: MetricsState[keyof MetricsState]) => string]> = [
  ["jobs_processed", "Processed", (value) => (value as number).toFixed(0)],
  ["jobs_completed", "Completed", (value) => (value as number).toFixed(0)],
  ["jobs_rejected", "Rejected", (value) => (value as number).toFixed(0)],
  ["sla_violations_total", "SLA Violations", (value) => (value as number).toFixed(0)],
  [
    "priority_weighted_completion_rate",
    "Weighted Completion",
    (value) => `${Math.round((value as number) * 100)}%`
  ],
  ["average_response_time", "Avg Response", (value) => (value as number).toFixed(2)],
  ["average_queue_length", "Avg Queue", (value) => (value as number).toFixed(2)],
  ["average_cpu_utilization", "Avg CPU", (value) => `${Math.round((value as number) * 100)}%`],
  ["average_memory_utilization", "Avg Memory", (value) => `${Math.round((value as number) * 100)}%`],
  ["current_reward", "Reward", (value) => (value as number).toFixed(1)],
  ["episode_reward", "Episode Reward", (value) => (value as number).toFixed(1)]
];

export function MetricsPanel({ metrics }: MetricsPanelProps) {
  return (
    <section className="grid gap-3 sm:grid-cols-3 xl:grid-cols-9">
      {metricMap.map(([key, label, format]) => (
        <div key={key} className="metric-tile min-h-20">
          <span>{label}</span>
          <strong>{format(metrics[key])}</strong>
        </div>
      ))}
    </section>
  );
}
```

- [ ] **Step 6: Create frontend/src/components/TierWaitPanel.tsx**

```typescript
interface TierWaitPanelProps {
  maxWaitByTier: Record<string, number>;
}

const tiers: Array<{ key: string; label: string; className: string }> = [
  { key: "gold", label: "Gold", className: "border-gold/40 bg-gold/10 text-gold" },
  { key: "silver", label: "Silver", className: "border-silver/40 bg-silver/10 text-silver" },
  { key: "bronze", label: "Bronze", className: "border-bronze/40 bg-bronze/10 text-bronze" }
];

export function TierWaitPanel({ maxWaitByTier }: TierWaitPanelProps) {
  return (
    <section className="panel">
      <div className="section-title">Max Wait Per Tier</div>
      <div className="mt-4 grid grid-cols-3 gap-3">
        {tiers.map((tier) => (
          <div key={tier.key} className={`rounded border p-3 text-center ${tier.className}`}>
            <div className="text-xs font-semibold uppercase">{tier.label}</div>
            <div className="mt-1 text-2xl font-bold">{maxWaitByTier[tier.key] ?? 0}</div>
            <div className="text-xs text-slate-500">ticks</div>
          </div>
        ))}
      </div>
    </section>
  );
}
```

- [ ] **Step 7: Wire TierWaitPanel into App.tsx**

Add the import:

```typescript
import { TierWaitPanel } from "../components/TierWaitPanel";
```

Add the panel to the layout, after `<ServerGrid ... />`:

```typescript
        <ServerGrid servers={state.servers} selectedServer={state.decision.assigned_server} />
        <TierWaitPanel maxWaitByTier={state.metrics.max_wait_by_tier} />
        <LiveCharts history={state.history} />
```

- [ ] **Step 8: Type-check and build**

```bash
cd frontend && npx tsc --noEmit && npm run build
```

Expected: no type errors, build succeeds.

- [ ] **Step 9: Commit**

```bash
git add frontend/src/components/ServerGrid.tsx frontend/src/components/IncomingJobPanel.tsx frontend/src/components/JobFlow.tsx frontend/src/components/Header.tsx frontend/src/components/MetricsPanel.tsx frontend/src/components/TierWaitPanel.tsx frontend/src/pages/App.tsx
git commit -m "feat: tier color-coding, SLA/priority KPIs, and live max-wait-per-tier panel"
```

---

## Task 19: Full unit test suite gate

**Files:** none (verification-only task).

- [ ] **Step 1: Run the entire test suite**

```bash
pytest -v
```

Expected: every test from Tasks 1–14 passes (roughly 45+ tests), zero failures. This is the explicit gate from the build prompt's Section 15: **do not proceed to PPO retraining until this is green.**

- [ ] **Step 2: If anything fails, fix it before continuing**

Do not skip failing tests or mark them `xfail` — trace the failure to the task that introduced the regression and fix it there.

---

## Task 20: Retrain PPO and tune reward weights against the three failure modes

**Files:**
- Modify: `configs/config_v1_5.yaml` (reward weights only, iteratively, as needed)

This task is executed by me directly (approved during brainstorming), not handed to a reviewer as a code diff.

- [ ] **Step 1: Run the full training job**

```bash
python -c "from rich.console import Console; from utils.config_v15 import load_config_v15, ensure_directories_v15; from rl.train import train_agent_v15; c=load_config_v15(); ensure_directories_v15(c); train_agent_v15(c, Console())"
```

- [ ] **Step 2: Inspect TensorBoard component curves and the monitor CSV**

```bash
tensorboard --logdir results/v1.5/tensorboard --port 6006 &
```

Check, in this order (Section 5.4's stated failure-mode priority):

1. **Starvation** — does Bronze max-wait (visible via the Rich console output during training, and reconstructable from an evaluation pass) stay under the 60-tick bound from Global Constraints? If not, increase `starvation_guard.bonus_per_tick_over` or lower `starvation_guard.threshold_ticks` — never raise Bronze's `tier_weight` (that blurs the priority signal).
2. **Defensive over-rejection** — is the agent rejecting jobs that had plenty of deadline slack? If `reward_components/sla_violation` dominates `reward_components/rejection` in magnitude, soften `sla_violation_penalty` relative to `rejection_penalty_base`.
3. **Reward-scale imbalance** — is `reward_components/balance` the largest-magnitude term in a typical tick? If so, lower `balance_bonus_weight` — completion-driven reward should dominate.

- [ ] **Step 3: If tuning is needed, adjust configs/config_v1_5.yaml's reward block and retrain**

Repeat Steps 1–2 until Bronze max-wait is bounded and no failure mode dominates, or until 5 tuning iterations are exhausted (reasonable budget) — whichever comes first. If the budget is exhausted without full success, stop and report the remaining gap clearly rather than continuing indefinitely.

- [ ] **Step 4: Commit the final tuned config (if it changed from Task 3's starting values)**

```bash
git add configs/config_v1_5.yaml
git commit -m "tune: adjust v1.5 reward weights against starvation/over-rejection/balance failure modes"
```

---

## Task 21: Full evaluation run, commit `results/v1.5/`, verify acceptance criteria

**Files:**
- Create: `results/v1.5/evaluation_episodes.csv`, `results/v1.5/evaluation_summary.csv`, and plot PNGs (all generated, not hand-written)

- [ ] **Step 1: Run the full evaluation across all 5 schedulers**

```bash
python -c "from rich.console import Console; from utils.config_v15 import load_config_v15; from rl.evaluate import evaluate_all_v15; evaluate_all_v15(load_config_v15(), 'models/v1.5/ppo_v15_final.zip', Console())"
```

- [ ] **Step 2: Check every acceptance-criteria bullet against the printed summary table and CSVs**

- Reward components logged separately in TensorBoard (Task 1/13) — confirm via the TensorBoard UI from Task 20.
- PPO retrained without crashing, Bronze max-wait under 60 ticks (Task 20) — read `max_wait_bronze` for the `PPO Agent` row in `evaluation_summary.csv`.
- PPO's `sla_violation_rate` is lower than `Priority-Aware Least Loaded`'s, on identical seeded workloads (both evaluated with `config.seed + episode` seeds, Task 14) — compare the two rows directly.
- `Least Loaded` (blind) and `Priority-Aware Least Loaded` both appear in the summary, for contrast.
- All unit tests pass (Task 19, already gated).
- Dashboard shows tier colors, SLA counter, live max-wait panel (Task 18) — spot-check by running `python main.py` and observing the dashboard.
- `results/v1.5/evaluation_episodes.csv` and `evaluation_summary.csv` contain rows for `Random`, `Round Robin`, `Least Loaded`, `Priority-Aware Least Loaded`, and `PPO Agent`.

- [ ] **Step 3: Commit the results**

```bash
git add results/v1.5/
git commit -m "chore: commit v1.5 evaluation results as the new baseline for v2"
```

- [ ] **Step 4: Report to the user**

Summarize, in plain language: whether PPO beat priority-aware LL on SLA violation rate (with the actual numbers), whether Bronze starvation stayed bounded (with the actual max-wait figure), and flag anything from Task 20's tuning loop that didn't fully resolve within budget.
