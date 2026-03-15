TBVision hosts a CXR tuberculosis classifier plus helpers for data prep, training, and inference.

## Project layout

- `tbvision/xraytb_net/`: data preprocessing, model definitions, training loops, checkpoints, and evaluation helpers. Use the per-component requirements when working with the AI stack.
- `tbvision/backend/`: FastAPI service that loads a checkpoint from `weights/` and exposes `/health` + `/predict` for frontend clients.
- `tbvision/frontend/`: placeholder for the UI layer—add React/Vue/Svelte apps or Python dashboards (Streamlit/Gradio) here alongside their own dependencies.

## Python dependencies

Install everything with the aggregator:

```bash
pip install -r requirements.txt
```

Or install only the component you need:

- `pip install -r tbvision/xraytb_net/requirements.txt` — training, evaluation, and inference utilities (torch, albumentations, etc.).
- `pip install -r tbvision/backend/requirements.txt` — FastAPI, uvicorn, and multipart support (the AI stack is still required when you actually run the backend).
- `pip install -r tbvision/frontend/requirements.txt` — add frontend-specific Python deps (e.g., Streamlit, Gradio) if applicable.

## Backend inference service

1. Ensure a trained checkpoint exists (e.g., `weights/xraytb_net.pth`). Set `TBVISION_CHECKPOINT` if you store it elsewhere.
2. Run the FastAPI server:

```bash
TBVISION_CHECKPOINT=weights/xraytb_net.pth \\
python -m uvicorn tbvision.main:app --host 0.0.0.0 --port 8000 --reload
```

A bit of structure:

- `tbvision/core`: shared configuration helpers, logging setup, and environment-bound defaults.
- `tbvision/backend/services`: image-decoding utilities plus the `ClassifierService` singleton that wraps `tbvision.xraytb_net.inference.ClassificationService`.
- `tbvision/backend/api`: routers and schemas for `/health`, `/predict`, and `/rag`.
- `tbvision/backend/knowledge`: a lightweight knowledge base that scores simple embeddings to power a retrieval endpoint (`/rag`).

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
python -m tbvision.xraytb_net.training.train_classifier --data-dir dataset --save-dir weights
python -m tbvision.xraytb_net.training.train_classifier --help  # for all CLI arguments
```

For inference/evaluation utilities, import from `tbvision.xraytb_net.inference` (e.g., `ClassificationService` and `evaluate_model`).

## Frontend

Place UI code inside `tbvision/frontend/`. The FastAPI `/predict` endpoint expects an image upload with optional metadata—use that contract to drive the visualization, patient summaries, or review tools you build on top of TBVision.
