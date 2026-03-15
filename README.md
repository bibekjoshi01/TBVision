TBVision hosts a CXR tuberculosis classifier plus helpers for data prep, training, and inference.

## Project layout

- `ai_model/`: data preprocessing, model definitions, training loops, checkpoints, and evaluation helpers. Use the per-component requirements when working with the AI stack.
- `backend/`: FastAPI service that loads a checkpoint from `ai_model/weights` and exposes `/health` + `/predict` for frontend clients.
- `frontend/`: placeholder for the UI layer—add React/Vue/Svelte apps or Python dashboards (Streamlit/Gradio) here alongside their own dependencies.

## Python dependencies

Install everything with the aggregator:

```bash
pip install -r requirements.txt
```

Or install only the component you need:

- `pip install -r ai_model/requirements.txt` — training, evaluation, and inference utilities (torch, albumentations, etc.).
- `pip install -r backend/requirements.txt` — FastAPI, uvicorn, and multipart support (the AI stack is still required when you actually run the backend).
- `pip install -r frontend/requirements.txt` — add frontend-specific Python deps (e.g., Streamlit, Gradio) if applicable.

## Backend inference service

1. Ensure a trained checkpoint exists (e.g., `weights/ensemble-densenet121_best.pth`).
2. Run the FastAPI server:

```bash
TBVISION_CHECKPOINT=weights/ensemble-densenet121_best.pth \\
uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
```

### API endpoints

- `GET /health` — Confirms the checkpoint path, device, and service health.
- `POST /predict` — Multipart/form-data request that must include:

  - `image`: an image file (`image/png`, `image/jpeg`, etc.).
  - `metadata` (optional): JSON string with auxiliary fields (e.g., patient ID, exam date).

Response payload: predicted label, per-class probabilities, raw logits, the label map, the submitted metadata, and the model configuration that produced the prediction.

## Training & evaluation

Use the AI package entry points directly:

```bash
python -m ai_model.training.train_classifier --data-dir dataset --save-dir weights
python -m ai_model.training.train_classifier --help  # for all CLI arguments
```

For inference/evaluation utilities, import from `ai_model.inference` (e.g., `ClassificationService` and `evaluate_model`).

## Frontend

Place UI code inside `frontend/`. The FastAPI `/predict` endpoint expects an image upload with optional metadata—use that contract to drive the visualization, patient summaries, or review tools you build on top of TBVision.
