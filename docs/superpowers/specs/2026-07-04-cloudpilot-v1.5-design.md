# CloudPilot v1.5 Design Spec

Status: Approved for planning
Source: `CloudPilot_v1.5_Build_Prompt.md` (author's build prompt), reconciled against actual v1 code in this repo.

## 1. Objective

Convert CloudPilot from a homogeneous, single-objective scheduling problem into a heterogeneous, multi-objective one, and prove PPO handles the added complexity better than a priority-aware heuristic.

**Done when:**
- Servers have different capacity tiers; jobs have priority tiers and SLA deadlines.
- Reward trades off priority-weighted completion against SLA violations, overload, rejection, and queue length.
- PPO is retrained on the new environment and measurably beats a priority-aware Least Loaded baseline on SLA violation rate, on identical seeded workloads.
- No priority tier is starved (bounded max wait, verified quantitatively).
- Results committed to `results/v1.5/` as the new baseline for v2.

## 2. Reconciliation with actual v1 code

The build prompt flagged three assumptions as **[ASSUMPTION — verify against your code]**. Verified against `environment/`, `schedulers/`, `metrics/`, `backend/api/`:

1. **Queue retry mechanism — assumption was wrong.** v1 has no global queue and no retry logic. Each `Server` holds its own `queue` list; `cloud_env.py` appends an unplaceable job to `selected_server.queue` and never touches it again — it is never retried, never completes, never freed. This build must construct the retry mechanism from scratch (see Section 5), not modify existing behavior.
2. **Reward is an unlogged single scalar — confirmed correct.** `step()` accumulates one float inline; no component breakdown exists anywhere.
3. **Homogeneous capacity, 2-field per-server observation — half correct.** Capacity is homogeneous and shared, confirmed. But the actual per-server block is 3 fields (`cpu_util`, `mem_util`, `queue_len_norm`), giving 15 dims total (4×3 + 3 job fields), not 16. The v1.5 target of 24 dims is achieved by dropping the per-server queue field in favor of one global queue field (consistent with the queue becoming global — see Section 5).

**Two additional deviations from the build prompt's literal instructions, decided during brainstorming:**

- **File layout.** The build prompt's checklist places new files under `backend/environment/`, `backend/schedulers/`. Those `backend/*` packages are empty compatibility stubs (docstring only) — all real logic lives in top-level `environment/`, `schedulers/`, `metrics/`, `rl/`, imported directly by `backend/api/simulation.py`. New v1.5 files go in the top-level packages instead.
- **No `tests/` directory or `pytest` dependency exists yet.** Both are new additions for this build.

## 3. Decisions made during brainstorming

### 3.1 Data model: full rename

Adopt the build prompt's dict-based schema exactly, everywhere:

```python
class PriorityTier(Enum):
    GOLD = "gold"
    SILVER = "silver"
    BRONZE = "bronze"

class Job:
    id: str
    requirements: dict[str, float]   # {"cpu": 2.0, "mem": 4.0}
    runtime: int
    arrival_tick: int
    priority_tier: PriorityTier
    deadline_ticks: int
    ticks_waited: int = 0
    start_tick: int | None = None
    completion_tick: int | None = None

class Server:
    id: str
    capacity: dict[str, float]   # {"cpu": 16.0, "mem": 64.0}
    used: dict[str, float]
    running_jobs: list[Job]
```

This replaces v1's flat named fields (`job_id`, `cpu_required`, `memory_required`, `cpu_capacity`, `current_cpu`, ...) everywhere they're threaded through:
- `environment/job.py`, `environment/server.py`, `environment/generator.py`, `environment/cloud_env.py`
- `schedulers/*.py`
- `backend/api/schemas.py` (`JobState`, `ServerState`, etc.)
- `frontend/src/types.ts` and any component reading these fields directly (`ServerGrid.tsx`, `IncomingJobPanel.tsx`, `JobFlow.tsx`, `Heatmap.tsx`, `DecisionFlow.tsx`)

Rationale: matches the build prompt's own stated goal — this is what makes GPU (v2.5) a drop-in later instead of a rework — and keeps the codebase in the shape the rest of the roadmap assumes.

### 3.2 Global queue replaces per-server queues

`CloudResourceEnv` owns one `self.queue: list[Job]`, not per-server queues. Each tick, in order:

1. Increment `ticks_waited` for every queued job.
2. Drop any job with `ticks_waited > deadline_ticks` — counted as **both** an SLA violation and a rejection. Never attempt to place an already-expired job.
3. Attempt to place remaining queued jobs against current free capacity, sorted by **priority-then-FIFO**: all Gold before any Silver before any Bronze; within a tier, earlier-enqueued first, tie-broken by job id for jobs enqueued on the same tick. For each queued job in that order, place it on the server that minimizes `mean(used[r] / capacity[r] for r in capacity)` (average utilization across resources, post-placement queue length no longer factors in since the queue is global, not per-server) among servers with free capacity to fit it; ties broken by lowest server id. This target-server rule is a fixed environment mechanism — identical regardless of which scheduler/agent is active — so all scheduler comparisons stay apples-to-apples. Only the newly-arriving job's placement (step 4) is ever agent-controlled.
4. Process the new incoming job: the active scheduler/agent picks a server.
   - If it fits: schedule immediately.
   - If it doesn't fit **but could fit on some other server**: auto-queue (enters the global queue, retried per step 3 on future ticks).
   - If it **cannot fit on any server** (larger than every server's capacity): reject immediately, do not queue — avoids the job sitting uselessly until SLA expiry (Section 12's oversized-Gold-job edge case).

### 3.3 Reward function

Implement exactly as specified in the build prompt (Section 5.4), with `normalized_std_dev(utils)` defined as `std(utils) / 0.5` clamped to `[0, 1]` (0.5 is the maximum possible std dev for utilizations bounded in `[0,1]`).

Log every term separately per tick (`completion`, `sla_violation`, `rejection`, `overload`, `queue`, `balance`, `starvation_bonus`), aggregated per episode alongside the total, and surfaced per-component in TensorBoard via a custom callback (not currently wired up in `rl/train.py` — needs `tensorboard_log=` on the PPO constructor plus a callback that pulls component values out of `info` and calls `self.logger.record(...)`).

This logging lands **first**, on the existing v1 reward, before any other change — nothing downstream is debuggable without it.

### 3.4 Config

New nested Pydantic models loaded from `configs/config_v1_5.yaml` (`ClusterConfig`, `ServerSpec`, `PriorityConfig`, `RewardConfig`, `NormalizationConfig`, composed into an `AppConfigV15`), separate from the existing flat `AppConfig`/`config.yaml`, which stays untouched for v1 reproducibility. Add a fail-fast validator (Pydantic model validator, raised at load time) that `priority.distribution` values sum to 1.0.

### 3.5 Observation space (24 dims, exact schema)

| Segment | Fields | Count |
|---|---|---|
| Per server (×4) | `capacity_cpu_norm`, `capacity_mem_norm`, `cpu_util`, `mem_util` | 16 |
| Queue (global) | `queue_length_norm` | 1 |
| Incoming job | `cpu_req_norm`, `mem_req_norm`, `runtime_norm` | 3 |
| Incoming job priority | one-hot `[is_gold, is_silver, is_bronze]` | 3 |
| Incoming job urgency | `deadline_ticks_norm` | 1 |

Normalization constants come from `config_v1_5.yaml`'s `normalization` block, never hardcoded (so v2's zone-local observations can reuse this logic).

### 3.6 Baseline scheduler

Add `schedulers/priority_least_loaded.py`: same core logic as `least_loaded.py` (lowest-utilization server that fits), but retries queued jobs in the same priority-then-FIFO order the environment enforces for PPO (Section 3.2 step 3). Do not modify `least_loaded.py` — it remains the priority-blind baseline for contrast. `random_scheduler.py` and `round_robin.py` need no logic changes, only confirmation they still behave correctly against heterogeneous capacities.

### 3.7 Metrics

Extend `metrics/metrics.py`'s aggregator with: SLA violation rate, priority-weighted completion rate, max wait per tier (Gold/Silver/Bronze tracked separately), and the mean of each reward component per episode. Written to `results/v1.5/*.csv`, keyed by scheduler name, in the same shape as v1's evaluation output so existing plotting code (`visualization/plots.py`) doesn't need rework.

### 3.8 Training execution

I run training and reward-weight tuning myself (via Bash), reading back TensorBoard/CSV metrics, iterating against the three failure modes in Section 5.4's stated priority order (starvation first, defensive over-rejection second, reward-scale imbalance third) until Section 13's acceptance criteria are met or a reasonable iteration budget is exhausted — flagging clearly if the budget is hit without full success.

### 3.9 Dashboard

Implemented last, after the backend is proven out:
- Tier color-coding on job cards and server queue displays (gold/silver/bronze accents).
- New KPI panel entries: live SLA violation counter, priority-weighted completion rate.
- New panel: max-wait-per-tier, updating live.
- WebSocket payload additions: `priority_tier`, `deadline_ticks`, `ticks_waited` on job objects; `sla_violations_total`, `priority_weighted_completion_rate`, `max_wait_by_tier` on the per-tick metrics payload.

## 4. Testing plan (Section 11, unchanged from build prompt)

New `tests/` directory (does not currently exist), add `pytest` to `requirements.txt`:

- `test_reward.py` — fixed tick events (one completion per tier, one SLA violation, one rejection, known overload) → assert computed reward matches hand-calculated expected value.
- `test_priority_sampling.py` — 10,000 jobs, fixed seed → observed tier proportions within tolerance of configured distribution.
- `test_sla_expiry.py` — job with `deadline_ticks=3` queued 4 ticks → dropped, counted as both SLA-violated and rejected, not silently placed on tick 4.
- `test_starvation_guard.py` — synthetic scenario with Bronze perpetually crowded out → starvation bonus increases and forces placement within a bounded number of extra ticks.
- Integration test — one full headless episode per scheduler (including PPO with an untrained/random policy) on the new heterogeneous+priority environment, no exceptions.

## 5. Edge cases (Section 12, unchanged from build prompt)

- Gold job larger than every server's capacity → rejected cleanly (Section 3.2 step 4), not looping until SLA expiry.
- Two same-tier jobs arriving same tick → deterministic FIFO tie-break by enqueue order then job id.
- All servers simultaneously at capacity when a Gold job arrives → environment doesn't crash, produces well-defined queue/reject outcome.
- Config sanity: `priority.distribution` must sum to 1.0, validated at load time (Section 3.4).

## 6. Acceptance criteria (Section 13, unchanged from build prompt)

- [ ] Reward components logged separately; TensorBoard shows per-component curves, not just the aggregate.
- [ ] PPO retrains without crashing and without a starvation blowup (Bronze max-wait bounded — concretely, under 3× the Bronze deadline range ceiling, i.e. under 60 ticks).
- [ ] PPO beats priority-aware Least Loaded on SLA violation rate, on identical seeded workloads.
- [ ] Priority-blind Least Loaded is retained and reported alongside the priority-aware version for contrast.
- [ ] All unit tests in Section 4 pass.
- [ ] Dashboard shows tier color-coding, SLA violation counter, and live max-wait-per-tier.
- [ ] `results/v1.5/*.csv` committed, containing all four schedulers (Random, Round Robin, blind-LL, priority-aware-LL) plus PPO.

## 7. File checklist (paths corrected to match actual project layout)

```
environment/
  heterogeneous.py       # NEW — server capacity classes, resource-dict model
  priority.py            # NEW — PriorityTier enum, SLA/deadline logic, starvation guard
  job.py, server.py, generator.py, cloud_env.py   # UPDATED — dict-based schema, 24-dim obs, global queue
schedulers/
  priority_least_loaded.py   # NEW
metrics/
  metrics.py              # UPDATED — SLA rate, priority-weighted completion, max-wait-per-tier
configs/
  config_v1_5.yaml         # NEW — config.yaml untouched
backend/api/
  schemas.py, simulation.py   # UPDATED — dict-based schema, new WS payload fields
frontend/src/
  types.ts, components/*  # UPDATED — dict-based schema, tier color-coding, new panels
results/
  v1.5/                    # NEW output directory
tests/
  test_reward.py, test_priority_sampling.py, test_sla_expiry.py, test_starvation_guard.py, test_integration.py   # NEW
requirements.txt           # UPDATED — add pytest
```

## 8. Implementation order (Section 15, unchanged from build prompt)

1. Reward component logging on the *existing* v1 reward.
2. Data model changes (dict-based Server/Job) with unit tests, no training yet.
3. Environment observation/reward rewiring, global queue, config file.
4. Priority-aware baseline.
5. Full unit test suite — don't train PPO until these pass.
6. Retrain PPO, watch component-level TensorBoard curves, tune weights against the three failure modes in priority order.
7. Metrics module + dashboard updates.
8. Full evaluation run, `results/v1.5/` committed, acceptance criteria checked off.
