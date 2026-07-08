"""FastAPI application for CloudPilot."""

from __future__ import annotations

import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.api.schemas import ControlRequest, SimulationState, TrainingState
from backend.api.simulation import LiveSimulation
from utils.helpers import load_config_v1_5

config = load_config_v1_5()
simulation = LiveSimulation(config)

app = FastAPI(title="CloudPilot API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup() -> None:
    """Start the live simulation when the backend boots."""
    await simulation.start()


@app.get("/api/state", response_model=SimulationState)
async def get_state() -> SimulationState:
    """Return the latest simulation state."""
    return await simulation.snapshot()


@app.post("/api/start", response_model=SimulationState)
async def start() -> SimulationState:
    """Resume simulation ticks."""
    await simulation.start()
    return await simulation.snapshot()


@app.post("/api/pause", response_model=SimulationState)
async def pause() -> SimulationState:
    """Pause simulation ticks."""
    await simulation.pause()
    return await simulation.snapshot()


@app.post("/api/reset", response_model=SimulationState)
async def reset() -> SimulationState:
    """Reset simulation state."""
    await simulation.reset()
    return await simulation.snapshot()


@app.post("/api/control", response_model=SimulationState)
async def control(request: ControlRequest) -> SimulationState:
    """Change scheduler or speed."""
    await simulation.configure(request.scheduler, request.speed)
    return await simulation.snapshot()


@app.get("/api/training", response_model=TrainingState)
async def training() -> TrainingState:
    """Return training progress placeholder for the dashboard."""
    return simulation.training


@app.websocket("/ws/simulation")
async def simulation_socket(websocket: WebSocket) -> None:
    """Stream simulation state to dashboard clients."""
    await websocket.accept()
    try:
        while True:
            state = await simulation.snapshot()
            await websocket.send_json(state.model_dump())
            await asyncio.sleep(0.25)
    except WebSocketDisconnect:
        return
