# Running the Full TBVision Stack

This guide shows how to launch both the backend API and the frontend portal simultaneously so you can run the full TBVision experience locally.

## Prerequisites

1. **Python & venv** – Python 3.11+ is required. Create/activate the `venv` in the repo root:

   Create virtual env:
   ```bash
   python -m venv venv
   ```

   Activate virtual env:
   For Unix/Linux/Mac:
   ```bash
   source venv/bin/activate
   ```

   For Windows:
   ```bash
   .\venv\Scripts\activate
   ```

2. **Dependencies** – Install backend and model requirements:

   ```bash
   pip install -r requirements.txt
   ```

3. **Node.js** – The frontend (under `frontend/`) expects Node 18+. Install deps when required.

4. **Model weights** – Ensure `weights/xraytb_net.pth` exists or point `CHECKPOINT_PATH` in `.env` to your checkpoint.

5. **Vector DB** – Start Qdrant (default `http://localhost:6333`). Create the collection manually if you customized it.
   - docker-compose up --build

6. **Load Knowledge base**

   ```bash
   python -m scripts.load_knowledge
   ```

7. **Environment variables** – Copy `.env.example` to `.env` with overrides such as:
   ```env
   CHECKPOINT_PATH=weights/xraytb_net.pth
   ALLOWED_ORIGINS=http://localhost:3000
   GEMINI_API_KEY=...
   MISTRAL_API_KEY=...
   ```

## Running the Full App

1. Start both backend and frontend together via the helper script (it launches the backend, then the frontend dev server with hot reload):

   ```bash
   ./run.sh
   ```

2. The backend listens on `http://127.0.0.1:8000` by default and exposes `/api/predict` + `/api/rag`. The frontend default port is `3000`.