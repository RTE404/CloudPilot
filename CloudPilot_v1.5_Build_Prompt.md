# CloudPilot v1.5 — Detailed Build Prompt

## 0. How to use this document

This is written to be precise enough to hand to a coding agent or follow yourself without re-deriving design decisions mid-implementation. Where it makes an assumption about how your current (v1) codebase works internally — since I've only seen the spec and README, not your actual source — it's flagged explicitly as **[ASSUMPTION — verify against your code]**. Confirm or correct those before implementing the section that depends on them.

---

## 1. Objective & Definition of Done

Convert CloudPilot from a homogeneous, single-objective scheduling problem into a heterogeneous, multi-objective one, and prove PPO handles the added complexity better than a priority-aware heuristic.

**v1.5 is done when:**
- Servers have different capacity tiers, jobs have priority tiers and SLA deadlines.
- The reward function trades off priority-weighted completion against SLA violations, overload, rejection, and queue length.
- PPO is retrained on the new environment and **measurably beats** a priority-aware Least Loaded baseline on SLA violation rate, on identical seeded workloads.
- No priority tier is starved (bounded max wait, verified quantitatively, not just eyeballed).
- Results are committed to `results/v1.5/` as the new baseline for v2.

---

## 2. Assumptions About Current (v1) Implementation — Verify Before Starting

**[ASSUMPTION]** The environment steps once per job arrival; the agent's action places *only the newly arrived job*. Jobs that can't be placed go into a waiting queue that is retried automatically (not agent-controlled) each tick as capacity frees up from completions, currently in **FIFO order**.

**[ASSUMPTION]** Reward is currently computed as a single scalar per step, not logged as separate components. If that's accurate, add component-level logging *before* starting the reward redesign below — you cannot debug a multi-objective reward you can only see as one number.

**[ASSUMPTION]** All 4 servers currently share one capacity (homogeneous), so the observation vector's per-server block is currently `[cpu_util, mem_util]` only (no capacity field, since it's constant and implicit). This build adds capacity as an explicit observation field, which changes the observation vector's shape — retraining from scratch is expected, not a bug to debug.

If any of these three don't match your actual code, adjust Sections 5 and 6 accordingly before implementing them.

---

## 3. Config Schema Changes

Add `configs/config_v1_5.yaml` (keep `config.yaml` for v1 reproducibility — don't overwrite it):

```yaml
cluster:
  servers:
    - { id: "A", class: "large",  cpu_capacity: 16, mem_capacity: 64 }
    - { id: "B", class: "medium", cpu_capacity: 8,  mem_capacity: 32 }
    - { id: "C", class: "medium", cpu_capacity: 8,  mem_capacity: 32 }
    - { id: "D", class: "small",  cpu_capacity: 4,  mem_capacity: 16 }

priority:
  distribution:            # sampling probability per incoming job
    gold: 0.15
    silver: 0.35
    bronze: 0.50
  deadline_ticks:           # ticks-until-must-start, sampled uniformly from range
    gold:   [3, 5]
    silver: [5, 10]
    bronze: [10, 20]
  queue_retry_order: "priority_fifo"   # gold > silver > bronze, FIFO within tier

reward:
  completion_base: 1.0
  tier_weights:
    gold: 3.0
    silver: 2.0
    bronze: 1.0
  sla_violation_penalty: -2.0          # multiplied by tier_weight
  overload_penalty_per_unit: -0.5      # per unit of resource over capacity
  rejection_penalty_base: -1.0         # multiplied by tier_weight
  queue_penalty_per_job_per_tick: -0.01
  balance_bonus_weight: 0.2
  starvation_guard:
    threshold_ticks: 15                # ticks waited before bonus kicks in
    bonus_per_tick_over: 0.05
    cap: 1.0

normalization:                          # for observation scaling — see Section 5.1
  max_job_cpu: 8
  max_job_mem: 32
  max_runtime: 50
  max_queue_length: 20
```

Every numeric value above is a **starting point, not a validated constant** — Section 5.4 tells you what to watch while tuning them.

---

## 4. Data Model Changes

### Server
Change from named attributes to a resource dict — this is what makes GPU (v2.5) a drop-in later instead of a rework.

```python
class Server:
    id: str
    capacity: dict[str, float]   # e.g. {"cpu": 16.0, "mem": 64.0}
    used: dict[str, float]       # same keys, current usage
    running_jobs: list[Job]
```

Utilization for resource `r` is `used[r] / capacity[r]`. Anywhere v1 code reads `server.cpu_capacity` directly, it now reads `server.capacity["cpu"]`.

### Job
Add three fields:

```python
class PriorityTier(Enum):
    GOLD = "gold"
    SILVER = "silver"
    BRONZE = "bronze"

class Job:
    id: str
    requirements: dict[str, float]   # e.g. {"cpu": 2.0, "mem": 4.0}
    runtime: int
    arrival_tick: int
    priority_tier: PriorityTier       # NEW
    deadline_ticks: int                # NEW — ticks-until-must-start, sampled at creation
    ticks_waited: int = 0               # NEW — incremented once per tick while queued
```

`ticks_waited` is what both the starvation guard and the SLA violation check key off of — make sure it increments exactly once per tick the job spends in the queue, not per placement attempt.

---

## 5. Environment Changes

### 5.1 Observation space (exact schema)

For a 4-server cluster, per tick:

| Segment | Fields | Count |
|---|---|---|
| Per server (×4) | `capacity_cpu_norm`, `capacity_mem_norm`, `cpu_util` (0–1), `mem_util` (0–1) | 16 |
| Queue | `queue_length_norm` | 1 |
| Incoming job | `cpu_req_norm`, `mem_req_norm`, `runtime_norm` | 3 |
| Incoming job priority | one-hot `[is_gold, is_silver, is_bronze]` | 3 |
| Incoming job urgency | `deadline_ticks_norm` | 1 |

**Total observation dimension: 24.**

Normalization: capacity fields divide by the max capacity present in the cluster config (so `16/16=1.0` for the large server); job fields divide by the `normalization.*` constants in config; deadline divides by the max value across all tiers' ranges. Keep normalization constants in config, not hardcoded, so v2's zone-local observations can reuse this logic.

### 5.2 Action space

Unchanged: `Discrete(4)` — pick one of the four servers for the incoming job. Heterogeneity lives entirely in capacity and reward, not in the action space.

### 5.3 Queue processing / retry semantics

Each tick, in this order:
1. Increment `ticks_waited` for every queued job.
2. **Check for SLA expiry first**: any queued job whose `ticks_waited > deadline_ticks` is dropped from the queue, counted as both an SLA violation and a rejection, and reward is penalized. Do not attempt to place an already-expired job — a job placed 1 tick late is still a completed job with a late start, whereas one that's been sitting past deadline is a clean failure. Keep those two cases distinguishable in your metrics.
3. Attempt to place remaining queued jobs against current free capacity, in **priority-then-FIFO order** (all Gold jobs before any Silver, all Silver before any Bronze, FIFO within a tier) — this is the `queue_retry_order` config value, and it's the mechanism, not the agent, that determines retry order. The agent only ever decides placement for the *newly arriving* job in step 4.
4. Process the new incoming job: agent selects a server (or the environment marks it rejected if the agent's chosen server can't fit it — decide now whether an infeasible action is auto-queued or auto-rejected, and keep it consistent, since it affects the rejection-penalty semantics).

### 5.4 Reward function

```python
def compute_reward(tick_events, servers, queue, cfg):
    r = 0.0

    for job in tick_events.completions:
        r += cfg.completion_base * cfg.tier_weights[job.priority_tier]

    for job in tick_events.sla_violations:
        r += cfg.sla_violation_penalty * cfg.tier_weights[job.priority_tier]

    for job in tick_events.rejections:
        r += cfg.rejection_penalty_base * cfg.tier_weights[job.priority_tier]

    total_overload_units = sum(
        max(0, used[r] - capacity[r]) for s in servers for r in s.capacity
    )
    r += cfg.overload_penalty_per_unit * total_overload_units

    r += cfg.queue_penalty_per_job_per_tick * len(queue)

    utils = [s.used[r] / s.capacity[r] for s in servers for r in s.capacity]
    r += cfg.balance_bonus_weight * (1 - normalized_std_dev(utils))

    for job in queue:
        if job.ticks_waited > cfg.starvation_guard.threshold_ticks:
            bonus = min(
                cfg.starvation_guard.bonus_per_tick_over
                * (job.ticks_waited - cfg.starvation_guard.threshold_ticks),
                cfg.starvation_guard.cap,
            )
            r += bonus  # this is a per-tick nudge toward eventually placing old jobs, not a placement reward itself

    return r
```

**What to watch while tuning these weights (in order of how often each failure mode actually shows up):**

1. **Priority starvation** — Bronze max-wait keeps climbing unbounded. Fix: increase the starvation bonus or lower its threshold, *not* the tier weights (raising Bronze's tier weight would blur the priority signal you're trying to teach).
2. **Defensive over-rejection** — if `sla_violation_penalty` is too harsh relative to `rejection_penalty_base`, the agent may learn to reject borderline jobs immediately rather than risk a late-SLA penalty later. Watch rejection rate for jobs that *could* have fit — if it's high for jobs with plenty of deadline slack, the penalty ratio is off.
3. **Reward scale imbalance** — if `balance_bonus_weight` dominates, the agent smooths utilization at the expense of actually completing high-priority jobs promptly. Check that completion-driven reward is the largest term in the *typical* per-tick reward breakdown, not the balance term.

### 5.5 Log reward components separately

Store each term (`completion`, `sla_violation`, `rejection`, `overload`, `queue`, `balance`, `starvation_bonus`) per tick, aggregated per episode, alongside the total. This is not optional instrumentation — every tuning decision above depends on being able to see which term moved.

---

## 6. Job Generator Changes

- Sample `priority_tier` per job from `priority.distribution`.
- Sample `deadline_ticks` uniformly from the range for that tier in `priority.deadline_ticks`.
- Keep the existing seeding mechanism exactly as-is — the tier and deadline sampling must be part of the same seeded RNG stream so `results/v1.5/` runs stay reproducible across scheduler comparisons.

---

## 7. Baseline Scheduler Changes

Add `backend/schedulers/priority_least_loaded.py`:

- Same core logic as Least Loaded (pick the server with lowest utilization that can fit the job), **except**: when choosing among multiple queued jobs to retry (Section 5.3), always process Gold before Silver before Bronze, FIFO within tier — same retry order the environment enforces for PPO, so the comparison is apples-to-apples.
- Do **not** modify the existing `least_loaded.py` — keep it as the priority-blind baseline. The contrast between blind-LL, priority-aware-LL, and PPO is a better result to report than PPO vs. only one baseline.
- `random.py` and `round_robin.py` need no logic changes, but confirm they still operate correctly against heterogeneous capacities (a job that doesn't fit anywhere should still be rejected/queued the same way it is for PPO).

---

## 8. Metrics Module Changes

Add to the aggregator:
- **SLA violation rate** — SLA violations / total jobs.
- **Priority-weighted completion rate** — Σ(tier_weight for completed jobs) / Σ(tier_weight for all jobs).
- **Max wait per tier** — track separately for Gold/Silver/Bronze; this is your starvation check, not a nice-to-have.
- **Reward component breakdown** — mean of each logged component (Section 5.5) per episode.

Write all of the above into `results/v1.5/*.csv`, keyed by scheduler name, same as v1's evaluation output format so existing plotting code doesn't need rework.

---

## 9. Training Changes

- Retrain PPO from scratch — the observation space shape changed (16→24 dims), so a v1 checkpoint is not fine-tunable into this, it's a fresh training run.
- Reward scale changed (multiple new penalty/bonus terms) — if you see training instability that didn't exist in v1, check total reward magnitude first; PPO is sensitive to reward scale, and a v1 learning-rate/clip-range that worked for the old reward range may need adjusting if the new composite reward's typical magnitude is much larger or smaller.
- Watch the TensorBoard reward curve component-by-component (Section 5.5's logging), not just the aggregate `ep_rew_mean` — a smooth aggregate curve can hide a starvation problem that only shows up in the Bronze max-wait metric.

---

## 10. Dashboard / Frontend Changes

- **Job cards and server queue displays**: color-code by tier (e.g., gold/silver/bronze accent colors) so tier is visually legible at a glance, not just in a tooltip.
- **New KPI panel entries**: SLA violation counter (live-incrementing), priority-weighted completion rate.
- **New panel**: max-wait-per-tier, updating live — this is the starvation check made visible, and it's a good thing to have on screen during a live demo since "look, Bronze isn't starving" is a strong point to make in an interview walkthrough.
- **WebSocket payload**: add `priority_tier`, `deadline_ticks`, `ticks_waited` to the job object already being streamed; add `sla_violations_total`, `priority_weighted_completion_rate`, `max_wait_by_tier` to the per-tick metrics payload.

---

## 11. Testing Plan

- **Unit test — reward calculation**: given a fixed set of tick events (one completion per tier, one SLA violation, one rejection, known overload), assert the computed reward matches a hand-calculated expected value. This is the single most valuable test in this milestone — reward bugs are otherwise invisible until training behaves strangely.
- **Unit test — priority sampling distribution**: generate 10,000 jobs with a fixed seed, assert observed tier proportions are within a reasonable tolerance of the configured distribution.
- **Unit test — SLA expiry logic**: a job with `deadline_ticks=3` that sits in queue for 4 ticks must be dropped and counted as both SLA-violated and rejected, not silently placed on tick 4.
- **Unit test — starvation guard**: a synthetic scenario where Bronze jobs are perpetually crowded out should show the starvation bonus increasing and eventually forcing placement — assert the job actually gets placed within a bounded number of extra ticks once the bonus is active.
- **Integration test**: run one full headless episode with each scheduler (including PPO with an untrained/random policy) end-to-end with no exceptions, on the new heterogeneous+priority environment.

---

## 12. Edge Cases & Failure Modes To Explicitly Verify

- A Gold job larger than every server's capacity (unschedulable regardless of policy) — confirm it's rejected cleanly rather than looping in the queue until it hits SLA expiry needlessly.
- Two jobs of the same tier arriving on the same tick — confirm FIFO tie-breaking is deterministic (by arrival order / job ID), since non-determinism here would break seeded-comparison reproducibility.
- All servers simultaneously at capacity when a Gold job arrives — confirm the environment doesn't crash and produces a well-defined rejection/queue outcome.
- Config sanity: `priority.distribution` values must sum to 1.0 — add a config validation check that fails fast (at startup, not mid-training) if they don't.

---

## 13. Acceptance Criteria

- [ ] Reward components logged separately; TensorBoard shows per-component curves, not just the aggregate.
- [ ] PPO retrains without crashing and without a starvation blowup (Bronze max-wait stays bounded — define "bounded" as a concrete number before you start, e.g. under 3× the Bronze deadline range ceiling).
- [ ] PPO beats priority-aware Least Loaded on SLA violation rate, on identical seeded workloads.
- [ ] Priority-blind Least Loaded is retained and reported alongside the priority-aware version for contrast.
- [ ] All unit tests in Section 11 pass.
- [ ] Dashboard shows tier color-coding, SLA violation counter, and live max-wait-per-tier.
- [ ] `results/v1.5/*.csv` committed, containing all four schedulers (Random, Round Robin, blind-LL, priority-aware-LL) plus PPO.

---

## 14. File Checklist

```
backend/environment/
  heterogeneous.py       # NEW — server capacity classes, resource-dict model
  priority.py            # NEW — PriorityTier enum, SLA/deadline logic, starvation guard
  (existing env files updated for 24-dim observation, resource-dict servers)
backend/schedulers/
  priority_least_loaded.py   # NEW
backend/metrics/
  (updated aggregator: SLA rate, priority-weighted completion, max-wait-per-tier)
configs/
  config_v1_5.yaml        # NEW — don't overwrite config.yaml
frontend/src/components/
  (updated: tier color-coding, SLA counter, max-wait panel)
results/
  v1.5/                   # NEW output directory
tests/
  test_reward.py          # NEW
  test_priority_sampling.py  # NEW
  test_sla_expiry.py      # NEW
  test_starvation_guard.py   # NEW
```

---

## 15. Suggested Implementation Order

1. Reward component logging on the *existing* v1 reward (before touching anything else — you need this instrumentation for every step after).
2. Data model changes (Server → resource dict, Job → priority/deadline fields) with unit tests, no training yet.
3. Environment observation/reward rewiring (Sections 5.1–5.4), config file, queue retry semantics.
4. Priority-aware baseline.
5. Full unit test suite (Section 11) — don't train PPO until these pass.
6. Retrain PPO, watch component-level TensorBoard curves, tune weights against the three failure modes in Section 5.4 in order.
7. Metrics module + dashboard updates.
8. Full evaluation run, `results/v1.5/` committed, acceptance criteria checked off.
