# Frontend Preview

This folder is reserved for the UI that will consume the FastAPI inference service.

- When you add a web app (React, Vue, Svelte, etc.), keep it under this directory.
- For simpler dashboards (Streamlit, Gradio), a standalone script with its own `requirements.*` file can live here.

Current expectations:
- POST `/predict` with `image` + optional JSON `metadata` (see root README).
- Display the returned `probabilities`, `prediction`, and any `metadata` fields.

Add any frontend-specific dependencies into `frontend/requirements.txt` or another platform-appropriate manifest (e.g., `package.json`).
