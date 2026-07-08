# CloudPilot v1.5 Project Status & Plan

## 1. What Are We Building
CloudPilot is an interactive cloud resource scheduling simulator powered by reinforcement learning. It currently supports a homogeneous, single-objective scheduling problem (v1.0). 

We are upgrading the project to **v1.5**. This upgrade converts CloudPilot into a **heterogeneous, multi-objective scheduling problem** to prove that Proximal Policy Optimization (PPO) handles added complexity better than priority-aware heuristics. 
Specifically, v1.5 introduces:
- Servers with different capacity tiers (large, medium, small).
- Incoming jobs with priority tiers (gold, silver, bronze) and Service Level Agreement (SLA) deadlines.
- A multi-objective reward function that trades off priority-weighted completions against SLA violations, overloads, rejections, and queue lengths.

## 2. Full Implementation Plan for v1.5
Based on the `CloudPilot_v1.5_Build_Prompt.md`, the implementation will follow these ordered phases:

### Phase 1: Instrumentation & Data Models
1. **Reward Component Logging**: Add separate logging for existing reward components (even on v1) to allow debugging and tuning later.
2. **Data Model Updates**:
   - `Server`: Refactor from hardcoded capacity attributes to a resource dictionary (`capacity`, `used`) to support heterogeneous resources.
   - `Job`: Add `priority_tier` (enum), `deadline_ticks`, and `ticks_waited`.

### Phase 2: Environment & Core Logic
3. **Observation & Environment Rewiring**:
   - Expand the observation space dimension from 16 to 24 to include capacity normalization, priority tiers (one-hot), and deadlines.
   - Create `config_v1_5.yaml` with server classes, priorities, deadlines, and reward weights.
   - Enforce SLA checks (dropping and penalizing expired jobs) and process queued jobs in Priority-then-FIFO order.
4. **Reward Function**: 
   - Implement the new composite reward function that handles priority-weighted completions, SLA penalties, rejection penalties, overload penalties, queue penalties, balance bonuses, and a starvation guard for low-priority jobs.

### Phase 3: Generators & Baselines
5. **Job Generator**:
   - Sample `priority_tier` and `deadline_ticks` for new jobs while preserving random seed reproducibility.
6. **Baselines**:
   - Add a new `priority_least_loaded.py` baseline to process queues in a priority-aware manner for comparison, while keeping the old priority-blind `least_loaded.py` unchanged.

### Phase 4: Testing & Training
7. **Unit Tests**:
   - Test reward calculation logic, priority sampling distribution, SLA expiry, and the starvation guard.
8. **PPO Training**:
   - Retrain the PPO model from scratch with the new 24-dimension observation space and monitor the reward components.

### Phase 5: Metrics & Dashboard
9. **Metrics Module**:
   - Track SLA violation rate, priority-weighted completion rate, and max wait per tier.
10. **Frontend Dashboard**:
    - Add color-coding for job tiers (e.g. gold, silver, bronze accents).
    - Add new KPI counters for SLA violations and max-wait panels to visualize starvation status live.
11. **Evaluation**:
    - Run evaluations across all schedulers (Random, Round Robin, Least Loaded, Priority Least Loaded, and PPO) and save results to `results/v1.5/`.

## 3. Work Done Till Now
- **v1.0 (MVP) is fully built and functional**, including the backend environment, FastAPI streaming, frontend React dashboard, and base heuristic schedulers.
- **No v1.5 specific changes have been implemented yet.** The codebase currently reflects a homogeneous environment with simple single-objective job queuing. 

## 4. Work Remaining
- **All steps from the v1.5 Implementation Plan (Phase 1 through Phase 5) are pending execution.**
- Start by implementing Phase 1 (Reward Component Logging and Data Model Updates) before moving on to environment rewiring and configuration changes.
