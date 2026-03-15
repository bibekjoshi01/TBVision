TBVision hosts a CXR tuberculosis classifier plus helpers for data prep, training, and inference.

## Project layout

- `xraytb_net/`: data preprocessing, model definitions, training loops, checkpoints, and evaluation helpers. Use the per-component requirements when working with the AI stack.
- `backend/`: FastAPI service that loads a checkpoint from `weights/` and exposes `/health` + `/predict` for frontend clients.
- `frontend/`: placeholder for the UI layer—add React/Vue/Svelte apps or Python dashboards (Streamlit/Gradio) here alongside their own dependencies.

## Python dependencies

Install everything with the aggregator:

```bash
pip install -r requirements.txt
```

Or install only the component you need:

- `pip install -r xraytb_net/requirements.txt` — training, evaluation, and inference utilities (torch, albumentations, etc.).
- `pip install -r backend/requirements.txt` — FastAPI, uvicorn, and multipart support (the AI stack is still required when you actually run the backend).
- `pip install -r frontend/requirements.txt` — add frontend-specific Python deps (e.g., Streamlit, Gradio) if applicable.

## Backend inference service

1. Ensure a trained checkpoint exists (e.g., `weights/ensemble-densenet121_best.pth`).
2. Run the FastAPI server:

```bash
TBVISION_CHECKPOINT=weights/ensemble-densenet121_best.pth \\
uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
```

A bit of structure:

- `backend/app/core`: configuration helpers, logging setup, and environment-bound defaults.
- `backend/app/services`: image-decoding utilities plus the `ClassifierService` singleton that wraps `xraytb_net.inference.ClassificationService`.
- `backend/app/api`: routers and schemas for `/health`, `/predict`, and `/rag`.
- `backend/app/rag`: a lightweight knowledge base that scores simple embeddings to power a retrieval endpoint (`/rag`).

### API endpoints

- `GET /health` — Reports whether the model is loaded, the checkpoint/device, and whether RAG is available.
- `POST /predict` — Multipart/form-data request that must include:

  - `image`: an image file (`image/png`, `image/jpeg`, etc.).
  - `metadata` (optional): JSON string with auxiliary fields (patient ID, exam date, etc.).

  Returns prediction, per-class probability distribution, raw logits, the label map, metadata echo, and the model config used.
- `POST /rag` — Queries the lightweight knowledge base with `{"question": "...", "top_k": 3}` and returns the most relevant docs that can be stitched into responses (useful for explainability or follow-up details).

## Training & evaluation

Use the AI package entry points directly:

```bash
python -m xraytb_net.training.train_classifier --data-dir dataset --save-dir weights
python -m xraytb_net.training.train_classifier --help  # for all CLI arguments
```

For inference/evaluation utilities, import from `xraytb_net.inference` (e.g., `ClassificationService` and `evaluate_model`).

## Frontend

Place UI code inside `frontend/`. The FastAPI `/predict` endpoint expects an image upload with optional metadata—use that contract to drive the visualization, patient summaries, or review tools you build on top of TBVision.
