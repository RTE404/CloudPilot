# CloudPilot Architecture

CloudPilot has three runtime layers:

1. The simulator layer in `environment/` implements the Gymnasium environment,
   server model, job model, and job generator.
2. The backend layer in `backend/` owns the live simulation loop, REST controls,
   and WebSocket state stream.
3. The frontend layer in `frontend/` renders the real-time operations dashboard.

All schedulers operate on `CloudResourceEnv`, which keeps benchmark behavior and
live dashboard behavior aligned.
