import type { SimulationState, TrainingState } from "../types";

const API_BASE = "";

async function post<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body)
  });
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export async function fetchState(): Promise<SimulationState> {
  const response = await fetch(`${API_BASE}/api/state`);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json() as Promise<SimulationState>;
}

export const startSimulation = () => post<SimulationState>("/api/start");
export const pauseSimulation = () => post<SimulationState>("/api/pause");
export const resetSimulation = () => post<SimulationState>("/api/reset");

export const updateControl = (body: {
  scheduler?: string;
  speed?: number;
}) => post<SimulationState>("/api/control", body);

export async function fetchTraining(): Promise<TrainingState> {
  const response = await fetch(`${API_BASE}/api/training`);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json() as Promise<TrainingState>;
}
