import { useEffect, useState } from "react";

import { fetchState } from "../services/api";
import type { SimulationState } from "../types";

export function useSimulation() {
  const [state, setState] = useState<SimulationState | null>(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetchState()
      .then((snapshot) => {
        if (!cancelled) {
          setState(snapshot);
        }
      })
      .catch(() => undefined);

    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(`${protocol}://${window.location.host}/ws/simulation`);

    socket.onopen = () => setConnected(true);
    socket.onclose = () => setConnected(false);
    socket.onerror = () => setConnected(false);
    socket.onmessage = (event) => {
      setState(JSON.parse(event.data) as SimulationState);
    };

    return () => {
      cancelled = true;
      socket.close();
    };
  }, []);

  return { state, connected };
}
