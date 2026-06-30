# CloudPilot

CloudPilot is an interactive cloud resource scheduling simulator powered by
reinforcement learning. It combines a Gymnasium cloud environment, PPO from
Stable-Baselines3, classical scheduling baselines, a FastAPI backend, and a
React + TypeScript live operations dashboard.

## What It Does

- Simulates a 4-server homogeneous cloud cluster.
- Generates one incoming job per simulation step.
- Assigns jobs using Random, Round Robin, Least Loaded, or PPO.
- Streams live server state, scheduling decisions, KPIs, charts, and heatmaps.
- Trains and evaluates a PPO scheduler against baseline algorithms.

## Architecture

```text
CloudPilot/
├── backend/          FastAPI app, WebSocket stream, live simulation service
├── environment/      Gymnasium cloud simulator, jobs, servers, generator
├── schedulers/       Random, round-robin, and least-loaded baselines
├── rl/               PPO model factory, training, and evaluation
├── metrics/          Evaluation metric aggregation
├── visualization/    Offline result plots
├── frontend/         React + TypeScript + TailwindCSS dashboard
├── configs/          YAML runtime configuration
├── models/           Trained PPO models
├── results/          Evaluation CSVs, logs, and plots
└── main.py           Local app launcher
```

## RL Formulation

The observation is a flattened vector containing each server's CPU utilization,
memory utilization, and queue length, plus the incoming job's CPU, memory, and
runtime requirements.

The action space is discrete:

- `0`: Server A
- `1`: Server B
- `2`: Server C
- `3`: Server D

The reward function gives positive feedback for successful scheduling, job
completion, and balanced utilization. It penalizes overload, rejection, and
long queues.

## Install

```bash
pip install -r requirements.txt
cd frontend
npm install
cd ..
```

## Run

```bash
python main.py
```

Open:

- Dashboard: `http://127.0.0.1:5173`
- Backend API: `http://127.0.0.1:8000`

Backend only:

```bash
python main.py --backend-only
```

Frontend only:

```bash
python main.py --frontend-only
```

## Dashboard

The dashboard includes:

- Header with simulation status, scheduler, episode, and speed.
- Incoming job panel with CPU, memory, and runtime.
- Four live server cards with utilization bars and running jobs.
- Live decision flow for the latest assignment.
- Animated job routing track.
- Live KPI panel.
- Reward, queue, utilization, completion, and rejection charts.
- Server utilization heatmap.
- Training progress panel.

## Training

The default PPO training configuration lives in `configs/config.yaml`.

```bash
python -c "from rich.console import Console; from utils.helpers import load_config, ensure_directories; from rl.train import train_agent; c=load_config(); ensure_directories(c); train_agent(c, Console())"
```

For a quick smoke run, temporarily lower `training_timesteps` in the config.

## Evaluation

After training, evaluate PPO against all baselines:

```bash
python -c "from rich.console import Console; from utils.helpers import load_config; from rl.evaluate import evaluate_all; evaluate_all(load_config(), 'models/ppo_final.zip', Console())"
```

Outputs are written to `results/`, including CSV summaries and plots.

## Configuration

Main settings are in `configs/config.yaml`, including server count, capacities,
episode length, job ranges, seed, simulation speed, training timesteps, and
evaluation episodes.

## Future Improvements

TODO:

- Heterogeneous servers
- Autoscaling
- Energy-aware scheduling
- GPU scheduling
- SLA-aware scheduling
- Priority queues
- Bursty traffic
- Multi-agent RL
- Kubernetes-inspired scheduling
- Container placement
- Edge computing simulation
