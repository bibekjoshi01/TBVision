# TBVision Backend

This component hosts the FastAPI-powered inference and retrieval services that sit on top of the `xraytb_net` model. It exposes clean endpoints for health checks, multi-class prediction, and retrieval-augmented responses. Use this document to install dependencies, configure runtime settings, and run the API locally.

## Repository layout

- `main.py`: FastAPI entrypoint that wires up configuration, logging, and the lifespan that loads `ClassifierService` and `RAGService` once at startup.
- `core/`: shared helpers (`config` loads `.env`, resolves checkpoint/rag document paths, validation) and `logging` configuration for structured logs.
- `services/`: adapters around the AI model and the lightweight document retrieval engine.
  - `classifier.py`: wraps `tbvision.xraytb_net.inference.ClassificationService`, checks the checkpoint, and keeps the prediction helpers typed.
  - `rag.py`: loads `knowledge/knowledge.json`, indexes documents via token-count embeddings, and serves top-`k` hits.
  - `image.py`: helpers for verifying `UploadFile` blobs and decoding them into NumPy arrays.
- `api/routers`: split HTTP routes for `/health`, `/predict`, and `/rag` plus Pydantic request/response schemas.
- `knowledge/knowledge.json`: answers used by the RAG endpoint when enabled.
- `xraytb_net/`: the classifier implementation; see `xraytb_net/README.md` for training and checkpoint export details.

## prerequisites

1. Python 3.11+ (the project already runs in a dedicated `venv`).
2. A model checkpoint available at `weights/xraytb_net.pth` (or override `CHECKPOINT_PATH`).
3. Optional: `knowledge/knowledge.json` to power the retrieval endpoint.

## Setup

````bash
python -m venv venv
source venv/bin/activate  # or `.
```bash powershell` on Windows
pip install -r requirements.txt
````

You can install extra model-related dependencies by activating the backend environment and running `pip install -r tbvision/xraytb_net/requirements.txt` if you plan to retrain or inspect the classifier implementation.

## Configuration

All runtime options can be set via `.env` files or environment variables because the Pydantic `Settings` object reads `app_env`, `checkpoint_path`, `rag_docs_path`, etc. Key overrides include:

- `CHECKPOINT_PATH`: Path to the `.pth` ensemble checkpoint (default `weights/xraytb_net.pth`).
- `RAG_DOCS_PATH`: Path to the JSON file that seeds the RAG knowledge base (default `knowledge/knowledge.json`).
- `APP_ENV`: Controls whether `/docs`/`/redoc` are exposed and toggles development defaults.
- `ALLOWED_ORIGINS`: Comma-separated list of origins permitted by CORS.
- `ENABLE_RAG`: Set to `false` to disable retrieval.

Example `.env` snippet:

```
CHECKPOINT_PATH=weights/xraytb_net.pth
APP_ENV=development
ENABLE_RAG=true
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173
```

After editing, restart Uvicorn so the lifespan handler reloads the services.

## Running the API

```bash
uvicorn tbvision.main:app --reload
```

- The app listens on `http://127.0.0.1:8000` by default. Add `--host 0.0.0.0 --port 9000` if you need to expose it externally.
- `--reload` is safe for local development but not production.
- Logs and fallback metadata are configured in `core/logging.py`.

## Available endpoints

### GET `/api/health`

Returns the service status, checkpoint path, device, and whether RAG is enabled.

### POST `/api/predict`

Content type: `multipart/form-data`

| Field      | Type   | Notes                                                                                      |
| ---------- | ------ | ------------------------------------------------------------------------------------------ |
| `image`    | `file` | Required image upload (JPEG/PNG).                                                          |
| `metadata` | `text` | Optional JSON string describing the sample. If provided, it must parse into a JSON object. |

Response (`PredictionResponse`):

- `prediction`: label string output from the ensemble.
- `probabilities`: class probability map.
- `raw_logits`: raw model logits for debugging.
- `label_map`: label ordering used by the classifier.
- `metadata`: echoed metadata (if provided).
- `config_snapshot`: runtime settings that produced the result (checkpoint path, image size, backbones, dropout, MC dropout flag).

### POST `/api/rag`

Body (`RAGRequest`):

```json
{
  "question": "How to interpret a cavitary lesion?",
  "top_k": 3
}
```

`top_k` is optional (default 3). The response includes the original question and the top matching documents (id, text, metadata, cosine-based similarity score).

## Model/service lifecycle

- The FastAPI lifespan handler (`main.py`) constructs a singleton `ClassifierService` and `RAGService` before the first request. Both must load successfully or the app refuses to start.
- `ClassifierService.load()` enforces that the checkpoint file exists and instantiates `ClassificationService` from `xraytb_net.inference`. Keep the weights directory synced with your training artifacts.
- `RAGService` gracefully handles missing docs (logs a warning and continues with the classifier-only API).

## Troubleshooting

- `ModuleNotFoundError: xraytb_net`: Ensure `tbvision/` is a package, the backend venv sees the repo root, and run commands from the repo root (the package name is `tbvision`).
- `RuntimeError: Checkpoint not found`: Place `xraytb_net.pth` under `weights/` or point `CHECKPOINT_PATH` at the correct file.
- `cv2` import errors: Activate the same virtualenv and reinstall `opencv-python`.

Refer to `tbvision/xraytb_net/README.md` for training, model exports, and evaluation. This backend README focuses on running the prediction/RAG HTTP service.
